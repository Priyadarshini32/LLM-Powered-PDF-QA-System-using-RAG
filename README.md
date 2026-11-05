# 📚 Academic PDF Question Answering System

A **Retrieval-Augmented Generation (RAG)** based academic question-answering chatbot that allows users to ask questions about the contents of a PDF document.

The application extracts text from the PDF, divides it into meaningful chunks, converts the chunks into vector embeddings, stores them in **ChromaDB**, retrieves the most relevant information for a user's question, and uses **Google Gemini** to generate a concise answer.

The application uses **LangGraph** to manage the conversation workflow and **Streamlit** to provide an interactive chat interface.

---

## 🚀 Features

* 📄 Question answering from a PDF document
* 🔎 Semantic similarity search
* 🧠 Retrieval-Augmented Generation (RAG)
* 🤖 Google Gemini for answer generation
* 🔢 Hugging Face multilingual embeddings
* 🗄️ ChromaDB vector database
* 🔗 LangGraph conversation workflow
* 💬 Streamlit chat interface
* 📑 Displays source PDF and reference page numbers
* 🧠 Maintains conversation history during the session
* 🌐 Supports multilingual embeddings
* ⚡ Cached models and vector database for better performance

---

## 🏗️ Architecture

```text
                    ┌─────────────────┐
                    │   PDF Document  │
                    │    data/*.pdf   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   PyPDFLoader   │
                    └────────┬────────┘
                             │
                             ▼
                  ┌─────────────────────┐
                  │ Recursive Text      │
                  │ Splitter            │
                  │                     │
                  │ Chunk Size: 1500    │
                  │ Overlap: 200        │
                  └──────────┬──────────┘
                             │
                             ▼
                ┌────────────────────────┐
                │ multilingual-e5-large │
                │ Embedding Model        │
                └───────────┬────────────┘
                            │
                            ▼
                     ┌────────────┐
                     │ ChromaDB   │
                     └─────┬──────┘
                           │
                           │
                  ┌────────▼────────┐
                  │ User Question   │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Similarity      │
                  │ Search          │
                  │ Top 4 Chunks    │
                  └────────┬────────┘
                           │
                           ▼
                 ┌──────────────────┐
                 │ Retrieved PDF    │
                 │ Context          │
                 └────────┬─────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  LangGraph  │
                   │  + Memory   │
                   └──────┬──────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Gemini 2.5      │
                 │ Flash           │
                 └────────┬────────┘
                          │
                          ▼
                 ┌─────────────────┐
                 │ Streamlit       │
                 │ Chat Interface  │
                 └─────────────────┘
```

---

## 🛠️ Technologies Used

| Technology            | Purpose                          |
| --------------------- | -------------------------------- |
| Python                | Application development          |
| Streamlit             | Web-based chat interface         |
| LangChain             | LLM and RAG components           |
| LangGraph             | Conversation workflow and memory |
| Google Gemini         | Large Language Model             |
| Hugging Face          | Text embedding model             |
| multilingual-e5-large | Document and query embeddings    |
| ChromaDB              | Vector database                  |
| PyPDF                 | PDF text extraction              |
| python-dotenv         | Environment variable management  |

---

## 📁 Project Structure

```text
Academic_QA/
│
├── data/
│   └── your_file.pdf
│
├── docs/
│   └── chroma/
│
├── .env
├── .gitignore
├── config.py
├── ingest.py
├── app.py
├── requirements.txt
└── README.md
```

### File Description

#### `app.py`

Main Streamlit application.

It:

* Loads the embedding model
* Loads ChromaDB
* Loads Gemini
* Retrieves relevant PDF chunks
* Runs the LangGraph workflow
* Generates answers
* Displays source information
* Maintains chat history

#### `ingest.py`

Responsible for creating the vector database.

It:

1. Finds the PDF in the `data` directory
2. Loads the PDF
3. Splits the text into chunks
4. Adds page metadata
5. Creates embeddings
6. Stores the vectors in ChromaDB

#### `config.py`

Contains application configuration such as:

* PDF directory
* ChromaDB directory
* Embedding model
* Chunk size
* Chunk overlap
* Retrieval count
* Gemini model

#### `.env`

Stores the Gemini API key securely.

#### `requirements.txt`

Contains the Python dependencies required by the project.

---

# ⚙️ Installation

## 1. Clone or download the project

Navigate to the project directory:

```powershell
cd Academic_QA
```

---

## 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

You should see:

```text
(.venv)
```

at the beginning of your terminal prompt.

---

## 3. Install dependencies

Run:

```powershell
pip install -r requirements.txt
```

---

# 🔑 Gemini API Configuration

Create a `.env` file in the project root:

```text
.env
```

Add your Gemini API key:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

Example:

```env
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXX
```

### Important

Do not commit your `.env` file to GitHub.

Add the following to `.gitignore`:

```text
.env
.venv/
__pycache__/
docs/chroma/
```

---

# 📄 Add Your PDF

Place your PDF inside the `data` directory:

```text
data/
└── academic_paper.pdf
```

The application automatically searches for a `.pdf` file inside `data`.

You do **not** need to specify the filename in the Python code.

### Current setup

This project is designed for **one PDF document**.

---

# 🧠 Create the Vector Database

Before starting the Streamlit application, the PDF must be converted into embeddings and stored in ChromaDB.

Run:

```powershell
python ingest.py
```

The ingestion process performs the following:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
ChromaDB
```

Example output:

```text
==========================================
       PDF VECTOR DATABASE CREATION
==========================================

PDF found:
data/academic_paper.pdf

Loading PDF...
PDF loaded successfully.
Total pages: 20

Splitting PDF into chunks...
Total chunks created: 35

Adding metadata...

Loading embedding model...
Model: intfloat/multilingual-e5-large

Embedding model loaded.

Creating ChromaDB...

==========================================
       VECTOR DATABASE CREATED
==========================================

PDF       : academic_paper.pdf
Pages     : 20
Chunks    : 35
Database  : docs/chroma

You can now run the Streamlit application:

streamlit run app.py

==========================================
```

---

# ▶️ Run the Application

After successfully creating the vector database, run:

```powershell
streamlit run app.py
```

Streamlit will provide a local URL such as:

```text
Local URL: http://localhost:8501
```

Open the URL in your browser.

---

# 💬 Using the Chatbot

Once the application is running, ask questions about the PDF.

For example:

```text
What is the main objective of the research?
```

```text
What methodology was used in the study?
```

```text
What are the major findings?
```

```text
What dataset was used?
```

```text
Explain the proposed methodology.
```

The application retrieves the most relevant sections of the PDF and sends them to Gemini.

---

# 📑 Source References

The application displays the source document and page numbers along with the generated answer.

Example:

```text
The proposed approach uses machine learning to...

---
📄 Source: academic_paper.pdf (Page 5), academic_paper.pdf (Page 7)
```

This makes it easier to verify the answer against the original document.

---

# 🔍 How RAG Works in This Project

The project follows the standard RAG pipeline.

## 1. Document Loading

`PyPDFLoader` extracts the text from the PDF.

```text
PDF → Text
```

---

## 2. Text Chunking

The extracted text is divided into smaller sections.

Current configuration:

```text
Chunk Size: 1500
Chunk Overlap: 200
```

The overlap helps preserve context between neighboring chunks.

---

## 3. Embeddings

Each chunk is converted into a numerical vector using:

```text
intfloat/multilingual-e5-large
```

These vectors represent the semantic meaning of the text.

---

## 4. Vector Storage

The embeddings are stored in:

```text
docs/chroma/
```

using ChromaDB.

---

## 5. User Query

When a user asks a question, the question is also converted into an embedding.

---

## 6. Similarity Search

ChromaDB finds the most semantically relevant chunks.

The application retrieves:

```text
Top 4 chunks
```

by default.

---

## 7. Context Construction

The retrieved chunks are combined into a context that contains:

* Source document
* Page number
* Extracted content

---

## 8. Gemini

The retrieved context and user's question are passed to:

```text
Gemini 2.5 Flash
```

Gemini generates the final answer based on the retrieved PDF information.

---

# 🧠 LangGraph Workflow

The application uses LangGraph to manage the conversation.

The workflow is:

```text
START
  │
  ▼
Model Node
  │
  ▼
Gemini
```

The `MemorySaver` checkpointer maintains conversation state during the session.

This allows users to ask follow-up questions such as:

```text
User:
What methodology was used?

Assistant:
The research uses ...

User:
Why was this methodology selected?

Assistant:
It was selected because...
```

---

# ⚙️ Configuration

The main configuration is stored in `config.py`.

### Embedding Model

```python
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
```

### Chunk Size

```python
CHUNK_SIZE = 1500
```

### Chunk Overlap

```python
CHUNK_OVERLAP = 200
```

### Number of Retrieved Chunks

```python
RETRIEVAL_K = 4
```

### Gemini Model

```python
GEMINI_MODEL = "gemini-3.6-flash"
```

### Temperature

```python
TEMPERATURE = 0.0
```

---

# 🔄 Updating the PDF

If you replace the PDF with another document, you must recreate the vector database.

For example:

```text
data/
└── new_document.pdf
```

Then run:

```powershell
python ingest.py
```

The existing ChromaDB is removed and recreated using the new PDF.

Then start the application:

```powershell
streamlit run app.py
```

---

# 🧪 Example Questions

You can test the application with questions such as:

### Basic Retrieval

```text
What is this document about?
```

### Specific Information

```text
What dataset was used?
```

### Methodology

```text
What methodology was followed?
```

### Results

```text
What are the main findings?
```

### Comparison

```text
What is the difference between the two approaches discussed?
```

### Follow-up Question

```text
Why is this approach better?
```

---

# 🛡️ Hallucination Control

The system prompt instructs Gemini to answer using the retrieved PDF context.

If the required information is not found, the chatbot responds:

```text
I couldn't find this information in the provided PDF.
```

This helps reduce unsupported answers.

However, RAG systems are not guaranteed to be completely hallucination-free, so important information should still be verified against the original document.

---

# ⚠️ Limitations

Currently, the application has the following limitations:

* Designed for one PDF
* Scanned/image-only PDFs require OCR
* Tables may not be extracted perfectly
* Complex PDF layouts may affect text extraction
* Conversation memory is in-memory and resets when the application restarts
* ChromaDB must be rebuilt when the PDF changes

---

# 🐛 Troubleshooting

## `GEMINI_API_KEY is missing`

Make sure `.env` exists in the project root:

```text
Academic_QA/
├── .env
├── app.py
└── ...
```

And contains:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

---

## `No PDF found`

Make sure your PDF is inside:

```text
data/
```

Example:

```text
data/
└── research.pdf
```

Then run:

```powershell
python ingest.py
```

---

## `ChromaDB has not been created`

Run:

```powershell
python ingest.py
```

before:

```powershell
streamlit run app.py
```

---

## Embedding model takes a long time

The first execution downloads:

```text
intfloat/multilingual-e5-large
```

This is expected.

After the model is downloaded, subsequent runs should be faster.

---

## PDF has no useful text

If the PDF is scanned, PyPDF may not be able to extract the text.

You will need to add an OCR pipeline such as:

```text
Scanned PDF
     ↓
OCR
     ↓
Extracted Text
     ↓
Chunking
     ↓
Embeddings
     ↓
ChromaDB
```

---

# 📦 Requirements

The main dependencies are:

```text
streamlit
python-dotenv

langchain
langchain-core
langchain-community
langchain-text-splitters

langgraph

langchain-google-genai

chromadb

pypdf
sentence-transformers

torch
transformers
huggingface-hub
```

Install them with:

```powershell
pip install -r requirements.txt
```

---

# 🔐 Security

Never expose your Gemini API key in source code.

### ❌ Do not do this

```python
GEMINI_API_KEY = "AIzaSyXXXXXXXX"
```

### ✅ Use `.env`

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

And load it using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

---

# 🚀 Quick Start

For a quick setup, run these commands:

```powershell
# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your Gemini API key to .env

# Put your PDF inside data/

# Create vector database
python ingest.py

# Start application
streamlit run app.py
```

---

# 📌 Project Summary

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline:

```text
                    PDF
                     │
                     ▼
                PDF Loader
                     │
                     ▼
               Text Chunking
                     │
                     ▼
               E5 Embeddings
                     │
                     ▼
                 ChromaDB
                     │
                     ▼
              User Question
                     │
                     ▼
             Similarity Search
                     │
                     ▼
            Relevant PDF Context
                     │
                     ▼
                LangGraph
                     │
                     ▼
             Gemini 2.5 Flash
                     │
                     ▼
               Final Answer
                     │
                     ▼
               Streamlit UI
```


