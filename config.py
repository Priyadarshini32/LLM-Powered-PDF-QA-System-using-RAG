import os


class Config:
    """
    Configuration class for the Academic PDF QA application.
    """

    # =========================================================
    # PDF CONFIGURATION
    # =========================================================

    # Folder containing the source PDF
    PDF_SOURCE_DIRECTORY = "data"

    # ChromaDB persistent storage
    CHROMA_PERSIST_DIRECTORY = "docs/chroma"

    # =========================================================
    # EMBEDDING CONFIGURATION
    # =========================================================

    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

    # -------------------------------------------------------------
    # SEMANTIC CHUNKING CONFIGURATION
    # -------------------------------------------------------------
    #
    # Semantic chunking groups sentences together based on the
    # similarity of their embeddings, starting a new chunk wherever
    # the topic shifts — instead of cutting at a fixed character
    # count.
    #
    # Breakpoint threshold strategies ("percentile",
    # "standard_deviation", "interquartile"):
    #
    #   percentile          – split when the distance between
    #                         consecutive sentence embeddings
    #                         exceeds this percentile.
    #                         Higher value => fewer, larger chunks.
    #
    #   standard_deviation  – split when the distance exceeds this
    #                         many standard deviations above the mean.
    #
    #   interquartile       – split using the interquartile range.
    #
    SEMANTIC_BREAKPOINT_TYPE = "percentile"

    # Threshold amount for the chosen breakpoint strategy
    # (for "percentile": 90 means "split at the largest 10%
    # of consecutive-sentence distances")
    SEMANTIC_BREAKPOINT_AMOUNT = 90

    # Hard upper bound (characters) for any single semantic chunk —
    # keeps very homogeneous sections from growing without limit.
    SEMANTIC_MAX_CHUNK_SIZE = 3000

    # Minimum size (characters) of a semantic chunk; smaller
    # fragments are merged forward to avoid tiny noisy chunks.
    SEMANTIC_MIN_CHUNK_SIZE = 350

    # ---------------------------------------------------------
    # Legacy fixed-size settings (kept for reference / rollback)
    # ---------------------------------------------------------
    # CHUNK_SIZE = 1500
    # CHUNK_OVERLAP = 200

    # Number of chunks to retrieve for each question
    RETRIEVAL_K = 4

    # =========================================================
    # GEMINI CONFIGURATION
    # =========================================================

    GEMINI_MODEL = "gemini-3.6-flash"

    TEMPERATURE = 0.0

    MAX_TOKENS = 2000

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(self):

        # Create directories if they don't exist
        os.makedirs(
            self.PDF_SOURCE_DIRECTORY,
            exist_ok=True
        )

        os.makedirs(
            self.CHROMA_PERSIST_DIRECTORY,
            exist_ok=True
        )


# Global configuration object
config = Config()