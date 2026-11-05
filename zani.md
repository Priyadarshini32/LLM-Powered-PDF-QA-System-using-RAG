# RAG Project — zani.md

Project memory maintained by Zani. Captures purpose, decisions, and a
change log so context carries across sessions.

## Overview
Academic PDF Question-Answering app (RAG). Pipeline: load a PDF from
`data/` → split into chunks → embed with a local HuggingFace model →
store in a persistent Chroma DB (`docs/chroma`) → answer questions in a
Streamlit UI (`app.py`) backed by Google Gemini.

Key files:
- `ingest.py` — ingestion pipeline (load PDF, chunk, embed, build Chroma).
- `app.py`    — Streamlit chat UI + retriever + Gemini LLM.
- `config.py` — central settings (paths, models, chunking, retrieval-k).
- `script.ipynb` — exploratory notebook.

## Decisions
- Embeddings: `intfloat/multilingual-e5-large` (local, sentence-transformers).
- Vector store: Chroma persisted at `docs/chroma`, collection `academic_documents`.
- LLM: Google Gemini (`GEMINI_MODEL`), temperature 0.

## Change log
- Switched chunking from FIXED-SIZE (`RecursiveCharacterTextSplitter`,
  1500/200) to SEMANTIC chunking (`langchain_experimental.SemanticChunker`)
  driven by sentence-embedding breakpoints.
  - New config keys: `SEMANTIC_BREAKPOINT_TYPE` ("percentile"),
    `SEMANTIC_BREAKPOINT_AMOUNT` (90), `SEMANTIC_MAX_CHUNK_SIZE` (3000),
    `SEMANTIC_MIN_CHUNK_SIZE` (350). Old `CHUNK_SIZE`/`CHUNK_OVERLAP`
    kept commented-out for rollback.
  - `ingest.py`: loads embeddings BEFORE splitting; merges tiny fragments
    forward and caps oversized chunks (recursive, no overlap); tags each
    chunk with `chunk_method="semantic"`; normalises 1-based page numbers.
  - Added dependency `langchain-experimental`.
  - NOTE: could NOT pip-install/run in this sandbox (offline). Compile-checked
    only. User must `pip install -r requirements.txt` and re-run `python ingest.py`
    to rebuild the vector DB before querying.
