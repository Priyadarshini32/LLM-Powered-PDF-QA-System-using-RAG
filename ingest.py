"""
Ingestion pipeline for the Academic PDF QA application.

Loads the source PDF, splits it into chunks using SEMANTIC CHUNKING
(groups sentences whose embeddings stay similar; starts a new chunk
wherever the topic shifts), embeds the chunks, and stores them in a
persistent Chroma vector database.

Replaces the previous fixed-size RecursiveCharacterTextSplitter
approach.
"""

import os
import shutil

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)
from langchain_experimental.text_splitter import (
    SemanticChunker,
)

from config import config


# ============================================================
# FIND PDF
# ============================================================

def find_pdf():
    """
    Find the PDF inside the data directory.
    """

    pdf_files = [
        file
        for file in os.listdir(
            config.PDF_SOURCE_DIRECTORY
        )
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        raise FileNotFoundError(
            f"No PDF found inside "
            f"'{config.PDF_SOURCE_DIRECTORY}'.\n"
            f"Please place your PDF inside the data folder."
        )

    if len(pdf_files) > 1:

        print(
            "\nWARNING: Multiple PDF files found."
        )

        print(
            f"Using: {pdf_files[0]}"
        )

    pdf_path = os.path.join(
        config.PDF_SOURCE_DIRECTORY,
        pdf_files[0]
    )

    return pdf_path


# ============================================================
# MERGE SMALL FRAGMENTS
# ============================================================

def merge_small_fragments(
    chunks,
    minimum_chunk_characters,
):
    """
    Merge chunks shorter than ``minimum_chunk_characters``
    characters into the NEXT chunk (same page).

    Semantic chunking occasionally emits very small fragments
    (isolated headers, captions, stray sentences). Merging them
    forward keeps retrieval contexts meaningful.

    Parameters
    ----------
    chunks : list[Document]
        Documents sorted in original reading order.
    minimum_chunk_characters : int
        Fragments strictly shorter than this are merged forward.

    Returns
    -------
    list[Document]
        Cleaned list of chunks.
    """

    if not chunks:
        return []

    cleaned = []
    pending_content = None
    pending_metadata = None

    for chunk in chunks:

        content = chunk.page_content.strip()
        metadata = dict(chunk.metadata)

        if pending_content is not None:

            # Attach the carried-over fragment to this chunk
            content = (
                pending_content + "\n\n" + content
            ).strip()

            # Preserve the earliest page seen
            if (
                pending_metadata.get("page")
                != metadata.get("page")
            ):
                metadata["pages"] = [
                    pending_metadata.get("page"),
                    metadata.get("page"),
                ]
            elif "pages" in pending_metadata:

                metadata["pages"] = (
                    pending_metadata["pages"]
                    + [metadata.get("page")]
                )

            pending_content = None
            pending_metadata = None

        if len(content) < minimum_chunk_characters:

            # Too small — carry it forward
            pending_content = content
            pending_metadata = metadata
            continue

        cleaned.append(
            Document(
                page_content=content,
                metadata=metadata,
            )
        )

    # Nothing follows the last fragment — append it as-is
    if pending_content:

        cleaned.append(
            Document(
                page_content=pending_content,
                metadata=pending_metadata,
            )
        )

    return cleaned


# ============================================================
# CAP OVERSIZED CHUNKS
# ============================================================

def cap_oversized_chunks(
    chunks,
    maximum_chunk_characters,
):
    """
    Enforce a hard upper bound on chunk length.

    Very homogeneous sections of a book can otherwise grow into
    extremely large semantic chunks. Oversized chunks are split
    recursively along natural boundaries (paragraphs, sentences),
    WITHOUT overlap, preserving the original metadata.

    Parameters
    ----------
    chunks : list[Document]
    maximum_chunk_characters : int

    Returns
    -------
    list[Document]
    """

    capped = []

    for chunk in chunks:

        if (
            len(chunk.page_content)
            <= maximum_chunk_characters
        ):
            capped.append(chunk)
            continue

        overflow_splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=maximum_chunk_characters,
                chunk_overlap=0,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

        pieces = overflow_splitter.split_text(
            chunk.page_content
        )

        for piece_index, piece in enumerate(pieces):

            piece_metadata = dict(chunk.metadata)
            piece_metadata["part"] = piece_index + 1

            capped.append(
                Document(
                    page_content=piece,
                    metadata=piece_metadata,
                )
            )

    return capped


# ============================================================
# CREATE VECTOR DATABASE
# ============================================================

def create_vector_database():

    print("\n==========================================")
    print("       PDF VECTOR DATABASE CREATION")
    print("             (semantic chunking)")
    print("==========================================")

    # --------------------------------------------------------
    # 1. Find PDF
    # --------------------------------------------------------

    pdf_path = find_pdf()

    print(
        f"\nPDF found:\n{pdf_path}"
    )

    # --------------------------------------------------------
    # 2. Load PDF
    # --------------------------------------------------------

    print("\nLoading PDF...")

    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    print(
        "PDF loaded successfully."
    )

    print(
        f"Total pages: {len(documents)}"
    )

    # --------------------------------------------------------
    # 3. Load embedding model
    #    (needed BEFORE splitting: semantic chunking compares
    #    sentence embeddings to decide where to cut)
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    print(
        f"Model: {config.EMBEDDING_MODEL_NAME}"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    print(
        "Embedding model loaded."
    )

    # --------------------------------------------------------
    # 4. Semantic chunking
    # --------------------------------------------------------

    print(
        "\nSplitting PDF semantically..."
    )

    print(
        f"Breakpoint strategy : "
        f"{config.SEMANTIC_BREAKPOINT_TYPE} "
        f"(amount={config.SEMANTIC_BREAKPOINT_AMOUNT})"
    )

    print(
        f"Size bounds         : "
        f"min={config.SEMANTIC_MIN_CHUNK_SIZE}, "
        f"max={config.SEMANTIC_MAX_CHUNK_SIZE} chars"
    )

    semantic_kwargs = {
        "breakpoint_threshold_type":
            config.SEMANTIC_BREAKPOINT_TYPE,
        "breakpoint_threshold_amount":
            float(config.SEMANTIC_BREAKPOINT_AMOUNT),
        "buffer_size": 1,
    }

    # Older releases of SemanticChunker do not accept
    # ``breakpoint_threshold_amount`` — fall back gracefully.
    try:

        semantic_splitter = SemanticChunker(
            embeddings,
            **semantic_kwargs,
        )

    except TypeError:

        semantic_kwargs.pop(
            "breakpoint_threshold_amount",
            None,
        )

        semantic_splitter = SemanticChunker(
            embeddings,
            **semantic_kwargs,
        )

    chunks = semantic_splitter.split_documents(
        documents
    )

    print(
        f"Raw semantic chunks : {len(chunks)}"
    )

    # --------------------------------------------------------
    # 5. Post-process chunk sizes
    # --------------------------------------------------------

    print(
        "\nPost-processing chunk sizes..."
    )

    chunks = merge_small_fragments(
        chunks,
        config.SEMANTIC_MIN_CHUNK_SIZE,
    )

    chunks = cap_oversized_chunks(
        chunks,
        config.SEMANTIC_MAX_CHUNK_SIZE,
    )

    print(
        f"Final chunks        : {len(chunks)}"
    )

    lengths = [
        len(c.page_content) for c in chunks
    ]

    if lengths:

        average_length = (
            sum(lengths) / len(lengths)
        )

        print(
            f"Length stats        : "
            f"avg={average_length:.0f}, "
            f"min={min(lengths)}, "
            f"max={max(lengths)} chars"
        )

    # --------------------------------------------------------
    # 6. Add metadata
    # --------------------------------------------------------

    print("\nAdding metadata...")

    for chunk in chunks:

        # PDF filename
        chunk.metadata["source"] = (
            os.path.basename(pdf_path)
        )

        # Chunking provenance
        chunk.metadata["chunk_method"] = "semantic"

        # Convert page number from 0-based to 1-based
        if "page" in chunk.metadata:

            chunk.metadata["page"] = (
                chunk.metadata["page"] + 1
            )

        # Normalise multi-page records
        if "pages" in chunk.metadata:

            chunk.metadata["pages"] = [
                page + 1
                for page in chunk.metadata["pages"]
            ]

    # --------------------------------------------------------
    # 7. Remove old Chroma database
    # --------------------------------------------------------

    if os.path.exists(
        config.CHROMA_PERSIST_DIRECTORY
    ):

        print(
            "\nRemoving existing ChromaDB..."
        )

        shutil.rmtree(
            config.CHROMA_PERSIST_DIRECTORY
        )

    os.makedirs(
        config.CHROMA_PERSIST_DIRECTORY,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 8. Create ChromaDB
    # --------------------------------------------------------

    print(
        "\nCreating ChromaDB..."
    )

    BATCH_SIZE = 50

    vectordb = Chroma(
        persist_directory=(
            config.CHROMA_PERSIST_DIRECTORY
        ),
        embedding_function=embeddings,
        collection_name="academic_documents",
    )

    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            total_chunks,
        )

        batch = chunks[start:end]

        vectordb.add_documents(batch)

        print(
            f"Embedding chunks: "
            f"[{end}/{total_chunks}] completed"
        )

    # --------------------------------------------------------
    # 9. Completion
    # --------------------------------------------------------

    print("\n==========================================")
    print("       VECTOR DATABASE CREATED")
    print("==========================================")

    print(
        f"PDF       : {os.path.basename(pdf_path)}"
    )

    print(
        f"Pages     : {len(documents)}"
    )

    print(
        f"Chunks    : {len(chunks)}"
    )

    print(
        f"Method    : semantic "
        f"({config.SEMANTIC_BREAKPOINT_TYPE}/"
        f"{config.SEMANTIC_BREAKPOINT_AMOUNT})"
    )

    print(
        f"Database  : "
        f"{config.CHROMA_PERSIST_DIRECTORY}"
    )

    print(
        "\nYou can now run the Streamlit application:"
    )

    print(
        "\nstreamlit run app.py"
    )

    print(
        "\n==========================================\n"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    create_vector_database()
