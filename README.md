# Academic PDF Question Answering System

A Retrieval-Augmented Generation (RAG) application that lets users ask questions about a PDF document. It extracts PDF text, creates semantic chunks, stores their embeddings in ChromaDB, retrieves relevant content for each question, and uses Google Gemini to generate answers grounded in the retrieved context.

The application provides a Streamlit chat interface and uses LangGraph with an in-memory checkpointer for conversation state.

## Features

- Ask questions about a PDF document.
- Semantic chunking based on sentence-embedding similarity.
- Store and retrieve document embeddings using ChromaDB.
- Generate answers using Google Gemini.
- Display the source PDF filename and page references.
- Maintain conversation state during the application session.
- Configure the embedding model, chunking parameters, retrieval count, and Gemini model in `config.py`.

## RAG Workflow

```text
PDF
 |
 v
PyPDFLoader
 |
 v
Semantic Chunking
 |
 v
Chunk Post-processing
(Merge small fragments and split oversized chunks)
 |
 v
Hugging Face Embeddings
 |
 v
ChromaDB
 |
 v
User Question
 |
 v
Similarity Search (Top 4)
 |
 v
Retrieved PDF Context
 |
 v
LangGraph + Gemini
 |
 v
Answer with Source References
 |
 v
Streamlit Chat Interface
```

## Semantic Chunking

This project uses LangChain's `SemanticChunker` rather than fixed-size chunking. It compares embeddings of consecutive sentences and creates chunk boundaries when the semantic distance crosses the configured breakpoint threshold. This helps group text that is semantically related instead of splitting the document at arbitrary character counts.

### Configuration

The settings are defined in `config.py`.

| Setting | Value | Purpose |
|---|---:|---|
| Embedding model | `intfloat/multilingual-e5-large` | Creates sentence and document embeddings |
| Breakpoint strategy | `percentile` | Selects semantic boundaries using a percentile threshold |
| Breakpoint amount | `90` | Configures the percentile threshold |
| Buffer size | `1` | Configures the surrounding sentence buffer used by `SemanticChunker` |
| Minimum chunk size | `350` characters | Merges short fragments forward |
| Maximum chunk size | `3000` characters | Splits chunks that exceed the limit |
| Retrieved chunks | `4` | Number of results requested for each question |

### Chunk post-processing

After semantic splitting, the ingestion pipeline applies two post-processing steps:

1. **Merge small fragments:** Chunks shorter than 350 characters are carried forward and combined with the next chunk. Page metadata is retained, including a `pages` list when the merged content spans different pages. A final fragment with no following chunk is kept as-is.
2. **Split oversized chunks:** Chunks longer than 3000 characters are split using `RecursiveCharacterTextSplitter`. It prefers paragraph, line, sentence, and word boundaries, in that order. This additional split uses no overlap. A `part` metadata field identifies each resulting piece.

Each final chunk receives metadata including:
- `source`: PDF filename.
- `chunk_method`: `semantic`.
- `page`: page number converted to 1-based numbering.
- `pages`: page references when the merged chunk spans multiple pages.
- `part`: sub-chunk number when an oversized chunk is split.

The chunks are embedded and stored in the ChromaDB collection named `academic_documents`.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Chat interface |
| LangChain | PDF loading, semantic chunking, embeddings, and vector store integration |
| LangGraph | Conversation workflow and checkpointing |
| Google Gemini | Answer generation |
| Hugging Face `intfloat/multilingual-e5-large` | Embedding model |
| ChromaDB | Persistent vector database |
| PyPDF | PDF text extraction |
| python-dotenv | Loading the Gemini API key from `.env` |

## Project Structure

```text
RAG_PROJECT/
├── data/
│   └── your_document.pdf
├── docs/
│   └── chroma/             # Generated vector database
├── myvenv/                 # Local virtual environment; not committed
├── .env                    # API key; not committed
├── .gitignore
├── app.py                  # Streamlit application and chat workflow
├── config.py               # Application and model configuration
├── ingest.py               # PDF ingestion and vector database creation
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.8 or later.
- A Google Gemini API key.
- Internet access on the first run to download the embedding model and install dependencies.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG_PROJECT
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
myvenv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Do not commit `.env` or publish your API key.

### 5. Add a PDF

Place one PDF file in the `data/` directory. The ingestion script automatically selects the first PDF it finds. If multiple PDFs are present, it prints a warning and uses the first one returned by the directory listing.

## Run the Application

### 1. Create the vector database

Run this whenever you add or replace the PDF:

```bash
python ingest.py
```

The script loads the PDF, creates semantic chunks, applies chunk-size post-processing, generates embeddings, and stores them in `docs/chroma/`. It removes the existing ChromaDB directory before creating a new database.

### 2. Start the chat application

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Asking Questions

Enter a question about the PDF in the chat input. The application:

1. Retrieves the four most similar chunks from ChromaDB.
2. Builds a context containing the retrieved text, source filename, and page number.
3. Sends the context and question through the LangGraph workflow to Gemini.
4. Displays the generated answer and source references.

The assistant is instructed to use the retrieved PDF context and to say when the requested information cannot be found in that context. This reduces unsupported answers but does not guarantee that hallucinations will never occur. Verify important answers against the original PDF.

## Configuration

The main settings are in `config.py`:

```python
PDF_SOURCE_DIRECTORY = "data"
CHROMA_PERSIST_DIRECTORY = "docs/chroma"

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

SEMANTIC_BREAKPOINT_TYPE = "percentile"
SEMANTIC_BREAKPOINT_AMOUNT = 90
SEMANTIC_MAX_CHUNK_SIZE = 3000
SEMANTIC_MIN_CHUNK_SIZE = 350

RETRIEVAL_K = 4

GEMINI_MODEL = "gemini-3.6-flash"
TEMPERATURE = 0.0
MAX_TOKENS = 2000
```

Change these values in `config.py` to adjust the application configuration.

## Limitations

- The ingestion script is designed to process one PDF at a time.
- If multiple PDFs are present, only the first one selected by the directory listing is processed.
- Scanned PDFs may require OCR because the current pipeline uses PDF text extraction.
- Complex layouts and tables may not be extracted accurately.
- The chat checkpointer uses in-memory storage; conversation state is not intended to persist across application restarts.
- Re-running ingestion deletes and recreates the existing ChromaDB directory.

## Troubleshooting

**`GEMINI_API_KEY is missing`**

Check that `.env` is in the project root and contains:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

**`No PDF found`**

Place a `.pdf` file inside the `data/` directory and run:

```bash
python ingest.py
```

**ChromaDB has not been created**

Run the ingestion script before starting Streamlit:

```bash
python ingest.py
streamlit run app.py
```

**The embedding model downloads on the first run**

The Hugging Face embedding model is downloaded when it is first loaded. This is expected.

## License

Add a license file if you intend to distribute this project publicly.
