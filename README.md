\# 🚀 DocIntel AI — LLM-Powered Enterprise Document Intelligence





\*\*DocIntel AI\*\* is an \*\*LLM-powered enterprise document intelligence and RAG platform\*\* built with Python, Streamlit, Qdrant, BGE-M3, BGE Reranker, and \*\*Qwen 2.5 7B\*\* running locally through Ollama.





It transforms unstructured PDF documents into \*\*searchable knowledge, structured data, and grounded AI answers with source-level citations\*\*.





\## ✨ Key Highlights





\- 🤖 \*\*LLM-powered RAG\*\* using Qwen 2.5 7B + Ollama

\- 🧠 \*\*Semantic search\*\* with BAAI/BGE-M3 embeddings

\- 🔎 \*\*Hybrid retrieval\*\* combining vector search and BM25

\- 🎯 \*\*Cross-encoder reranking\*\* with BGE Reranker

\- 📚 \*\*Source and page-level citations\*\*

\- 🧩 \*\*Section-aware document parsing and chunking\*\*

\- 🏢 \*\*LLM-based structured document extraction\*\*

\- ✅ \*\*Pydantic schema validation\*\*

\- 📊 \*\*JSON and Excel export\*\*

\- 🔬 \*\*Retrieval evaluation and debugging tools\*\*

\- 💻 \*\*Automatic technical skill classification\*\*

\- 🔒 \*\*Local AI inference through Ollama\*\*





\## 🏗️ Architecture





```text

&#x20;                   PDF DOCUMENT

&#x20;                        │

&#x20;                        ▼

&#x20;                PyMuPDF Extraction

&#x20;                        │

&#x20;                        ▼

&#x20;             Section-Aware Chunking

&#x20;                        │

&#x20;             ┌──────────┴──────────┐

&#x20;             ▼                     ▼

&#x20;       BGE-M3 Embeddings          BM25

&#x20;             │                     │

&#x20;             └──────────┬──────────┘

&#x20;                        ▼

&#x20;                Hybrid RRF Retrieval

&#x20;                        │

&#x20;                        ▼

&#x20;                 BGE Reranker

&#x20;                        │

&#x20;                        ▼

&#x20;              Relevant Context

&#x20;                        │

&#x20;                        ▼

&#x20;                Qwen 2.5 7B

&#x20;                 via Ollama

&#x20;                        │

&#x20;                        ▼

&#x20;            Grounded Answer + Citations

🔍 RAG Pipeline



DocIntel AI implements a complete Retrieval-Augmented Generation pipeline:



PDF text extraction with PyMuPDF

Section-aware document chunking

BGE-M3 semantic embeddings

BM25 lexical retrieval

Hybrid Reciprocal Rank Fusion

BGE cross-encoder reranking

Context-grounded Qwen generation

Document and page-level citations

RRF Configuration

RRF score = 1 / (k + rank)





RRF\_K = 60



Lexical-overlap and section-aware boosts are also applied to improve retrieval for exact technical terms and document-specific queries.



🤖 LLM Stack

Component	Technology

LLM	Qwen 2.5 7B

LLM Runtime	Ollama

Embeddings	BAAI/BGE-M3

Reranker	BAAI/BGE-Reranker-v2-m3

Vector Database	Qdrant

Lexical Search	BM25

Validation	Pydantic

Interface	Streamlit



The system is designed around local AI inference, making it suitable for privacy-sensitive document intelligence workflows.



🏢 Structured Document Intelligence



DocIntel AI includes an LLM-based structured extraction pipeline for CVs and similar documents.



PDF

&#x20;│

&#x20;▼

Text Extraction

&#x20;│

&#x20;▼

Qwen Structured Extraction

&#x20;│

&#x20;▼

Structured JSON

&#x20;│

&#x20;▼

Pydantic Validation

&#x20;│

&#x20;▼

Post-Processing

&#x20;│

&#x20;├── JSON Export

&#x20;└── Excel Export



The extraction schema includes:



Name, email, phone, location

GitHub and LinkedIn

Professional profile

Programming languages

ML frameworks

ML tools

ML methods

Data-science tools

Professional experience

Education

Publications

Human languages



Post-processing rules help prevent technologies such as NumPy, pandas, PyTorch, Git, Docker, and MLflow from being incorrectly classified as programming languages.



📚 Grounded Citations



Each retrieved chunk contains metadata:



Document

Page

Chunk

Section



Example:



\[1] resume.pdf | Page: 2 | Chunk: 3 | Section: experience



This allows generated answers to be traced back to the original document context.



🔬 Evaluation



The project includes a lightweight retrieval smoke test.



Example question:



What programming languages does the candidate know?



Expected keywords:



Python

SQL

Bash

MATLAB

Fortran



The evaluation reports:



Keyword Recall

Keywords Found

Keywords Expected



This is a lightweight pipeline validation test rather than a production benchmark.



🖥️ Application Workflow

Upload PDF

&#x20;   ↓

Extract \& Index Document

&#x20;   ↓

Document Statistics

&#x20;   ↓

User Question

&#x20;   ↓

Hybrid Retrieval

&#x20;   ↓

BGE Reranking

&#x20;   ↓

Qwen LLM Generation

&#x20;   ↓

Answer + Citations



Users can also extract structured CV/document information and export the results as JSON or Excel.



📦 Installation

pip install streamlit pymupdf pandas requests pydantic sentence-transformers rank-bm25 qdrant-client openpyxl

🤖 Ollama Setup



Download the Qwen model:



ollama pull qwen2.5:7b



Start Ollama:



ollama serve



The application expects:



http://localhost:11434/api/generate

▶️ Run the Application



Save the application as:



app.py



Run:



python -m streamlit run app.py



Then open the Streamlit URL shown in the terminal.



⚙️ Configuration

COLLECTION\_NAME = "docintel\_documents"





EMBEDDING\_MODEL\_NAME = "BAAI/bge-m3"





RERANKER\_MODEL\_NAME = "BAAI/bge-reranker-v2-m3"





VECTOR\_SIZE = 1024





OLLAMA\_URL = "http://localhost:11434/api/generate"





OLLAMA\_MODEL = "qwen2.5:7b"





RRF\_K = 60





CHUNK\_SIZE = 1200

CHUNK\_OVERLAP = 150

📂 Project Structure

docintel-ai/

├── app.py

├── README.md

├── requirements.txt

└── .gitignore

🛠️ Tech Stack



Python · Streamlit · PyMuPDF · Qdrant · Sentence Transformers · BGE-M3 · BGE Reranker · BM25 · Qwen 2.5 · Ollama · Pydantic · Pandas · OpenPyXL



⚠️ Current Limitations

Qdrant currently uses in-memory storage, so vectors are lost after application restart.

Scanned/image-only PDFs require OCR.

Embedding and reranking models can require significant RAM/VRAM.

Ollama must be running locally for LLM generation and structured extraction.

🚀 Enterprise Roadmap



Potential production improvements include:



Authentication and role-based access control

Multi-user document isolation

Persistent Qdrant deployment

Document-level permissions

OCR and table extraction

PII detection and protection

Audit logging

Background document processing

Retrieval and generation evaluation datasets

Monitoring and observability

Model/version management

Retry and failure handling

Production-grade security

💡 Project Highlights



DocIntel AI demonstrates an end-to-end LLM engineering and Document AI pipeline:



Document Processing

&#x20;       ↓

Embeddings

&#x20;       ↓

Vector Search + BM25

&#x20;       ↓

Hybrid Retrieval

&#x20;       ↓

Reranking

&#x20;       ↓

RAG

&#x20;       ↓

Local LLM

&#x20;       ↓

Structured Extraction

&#x20;       ↓

Pydantic Validation

&#x20;       ↓

JSON / Excel Export



The project showcases practical implementation of LLMs, RAG, semantic search, hybrid retrieval, reranking, NLP, information extraction, vector databases, structured generation, and local AI inference.



👤 Author



Naeim Mousavi



DocIntel AI — Enterprise Document Intelligence Engine



© 2026 Naeim Mousavi



📄 License



This project is licensed under the MIT License.



You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the conditions of the MIT License.



See the LICENSE file for the complete license text.



