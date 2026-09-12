# 🚀 DocIntel AI

## LLM-Powered Enterprise Document Intelligence & RAG

**DocIntel AI** is an **LLM-powered enterprise document intelligence and Retrieval-Augmented Generation (RAG) platform** built with Python, Streamlit, Qdrant, BGE-M3, BGE Reranker, and **Qwen 2.5 7B** running locally through Ollama.

The system transforms unstructured PDF documents into **searchable knowledge, structured information, and grounded AI answers with document- and page-level citations**.

---

# ✨ Key Highlights

* 🤖 **LLM-powered RAG** using Qwen 2.5 7B + Ollama
* 🧠 **Semantic search** using BAAI/BGE-M3 embeddings
* 🔎 **Hybrid retrieval** combining vector search and BM25
* 🎯 **Cross-encoder reranking** using BGE Reranker
* 📚 **Source- and page-level citations**
* 🧩 **Section-aware document parsing and chunking**
* 🏢 **LLM-based structured document extraction**
* ✅ **Pydantic schema validation**
* 📊 **JSON and Excel export**
* 🔬 **Retrieval evaluation and debugging tools**
* 💻 **Automatic technical skill classification**
* 🔒 **Local AI inference through Ollama**

---

# 🏗️ Architecture

```text
                         PDF DOCUMENT
                              │
                              ▼
                     PyMuPDF Extraction
                              │
                              ▼
                  Section-Aware Chunking
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
              BGE-M3 Embeddings       BM25
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    Hybrid RRF Retrieval
                              │
                              ▼
                       BGE Reranker
                              │
                              ▼
                    Relevant Context
                              │
                              ▼
                       Qwen 2.5 7B
                        via Ollama
                              │
                              ▼
                 Grounded Answer + Citations
```

---

# 🔍 RAG Pipeline

DocIntel AI implements an end-to-end **Retrieval-Augmented Generation** pipeline.

The workflow consists of:

1. PDF text extraction using PyMuPDF
2. Section-aware document chunking
3. BGE-M3 semantic embeddings
4. BM25 lexical retrieval
5. Hybrid Reciprocal Rank Fusion
6. BGE cross-encoder reranking
7. Context-grounded Qwen generation
8. Document- and page-level citations

### Retrieval Pipeline

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Section Detection
 │
 ▼
Chunking
 │
 ├───────────────┐
 ▼               ▼
BGE-M3          BM25
 │               │
 └───────┬───────┘
         ▼
     RRF Fusion
         │
         ▼
    BGE Reranker
         │
         ▼
 Relevant Context
         │
         ▼
    Qwen 2.5 7B
```

## Reciprocal Rank Fusion

The hybrid retrieval stage uses Reciprocal Rank Fusion (RRF).

The RRF score is defined as:

$$
RRF(d) = \frac{1}{k + rank(d)}
$$

The configured value is:

```python
RRF_K = 60
```

Lexical-overlap and section-aware boosts are also applied to improve retrieval for exact technical terminology and document-specific queries.

---

# 🤖 LLM Stack

| Component       | Technology              |
| --------------- | ----------------------- |
| LLM             | Qwen 2.5 7B             |
| LLM Runtime     | Ollama                  |
| Embeddings      | BAAI/BGE-M3             |
| Reranker        | BAAI/BGE-Reranker-v2-m3 |
| Vector Database | Qdrant                  |
| Lexical Search  | BM25                    |
| Validation      | Pydantic                |
| Interface       | Streamlit               |

The system is designed around **local AI inference**, making it suitable for privacy-sensitive document intelligence workflows where documents and inference should remain on the local machine.

---

# 🏢 Structured Document Intelligence

DocIntel AI includes an LLM-based structured extraction pipeline for CVs and similar documents.

```text
PDF
 │
 ▼
Text Extraction
 │
 ▼
Qwen Structured Extraction
 │
 ▼
Structured JSON
 │
 ▼
Pydantic Validation
 │
 ▼
Post-Processing
 │
 ├── JSON Export
 └── Excel Export
```

## Extraction Schema

The extraction pipeline can identify information such as:

* Name, email, phone, and location
* GitHub and LinkedIn
* Professional profile
* Programming languages
* Machine-learning frameworks
* Machine-learning tools
* Machine-learning methods
* Data-science tools
* Professional experience
* Education
* Publications
* Human languages

### Technical Skill Classification

Post-processing rules help prevent technologies such as:

* NumPy
* pandas
* PyTorch
* Git
* Docker
* MLflow

from being incorrectly classified as programming languages.

---

# 📚 Grounded Citations

Each retrieved document chunk contains metadata such as:

* Document
* Page
* Chunk
* Section

Example:

```text
[1] resume.pdf | Page: 2 | Chunk: 3 | Section: experience
```

This metadata allows generated answers to be traced back to the original document context.

The goal is to make the generated response **grounded and traceable rather than relying only on the model's internal knowledge**.

---

# 🔬 Evaluation

The project includes a lightweight **retrieval smoke test** for validating the retrieval pipeline.

### Example Question

```text
What programming languages does the candidate know?
```

### Expected Keywords

```text
Python
SQL
Bash
MATLAB
Fortran
```

The evaluation reports:

* Keyword Recall
* Keywords Found
* Keywords Expected

This is intended as a **lightweight pipeline validation test**, not as a production-scale retrieval benchmark.

---

# 🖥️ Application Workflow

The Streamlit application follows this workflow:

```text
Upload PDF
    │
    ▼
Extract & Index Document
    │
    ▼
Document Statistics
    │
    ▼
User Question
    │
    ▼
Hybrid Retrieval
    │
    ▼
BGE Reranking
    │
    ▼
Qwen LLM Generation
    │
    ▼
Answer + Citations
```

Users can also extract structured CV/document information and export the results as:

* JSON
* Excel

---

# 📦 Installation

Install the required Python packages:

```bash
pip install streamlit pymupdf pandas requests pydantic sentence-transformers rank-bm25 qdrant-client openpyxl
```

Alternatively, if the repository contains a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

# 🤖 Ollama Setup

DocIntel AI uses **Ollama** to run Qwen locally.

Pull the Qwen model:

```bash
ollama pull qwen2.5:7b
```

Start the Ollama service:

```bash
ollama serve
```

The application expects the Ollama generation API at:

```text
http://localhost:11434/api/generate
```

---

# ▶️ Run the Application

The main application is:

```text
app.py
```

Run Streamlit with:

```bash
python -m streamlit run app.py
```

Then open the Streamlit URL displayed in the terminal.

---

# ⚙️ Configuration

The main configuration values are:

```python
COLLECTION_NAME = "docintel_documents"

EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

VECTOR_SIZE = 1024

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "qwen2.5:7b"

RRF_K = 60

CHUNK_SIZE = 1200

CHUNK_OVERLAP = 150
```

These parameters control the vector collection, embedding model, reranker, LLM endpoint, retrieval configuration, and document chunking behavior.

---

# 📂 Project Structure

```text
docintel-ai/
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Technology Stack

| Category          | Technology   |
| ----------------- | ------------ |
| Language          | Python       |
| Interface         | Streamlit    |
| PDF Processing    | PyMuPDF      |
| Vector Database   | Qdrant       |
| Embeddings        | BGE-M3       |
| Reranking         | BGE Reranker |
| Lexical Retrieval | BM25         |
| LLM               | Qwen 2.5 7B  |
| Local Inference   | Ollama       |
| Validation        | Pydantic     |
| Data Processing   | Pandas       |
| Excel Export      | OpenPyXL     |

---

# ⚠️ Current Limitations

The current implementation has several limitations:

* Qdrant currently uses **in-memory storage**, so vectors are lost after application restart.
* Scanned or image-only PDFs require an OCR pipeline.
* Embedding and reranking models can require significant RAM/VRAM.
* Ollama must be running locally for LLM generation and structured extraction.
* The current retrieval evaluation is a lightweight smoke test rather than a comprehensive benchmark.
* Production deployment would require additional authentication, security, monitoring, and persistence mechanisms.

---

# 🚀 Enterprise Roadmap

Potential production improvements include:

* Authentication and role-based access control
* Multi-user document isolation
* Persistent Qdrant deployment
* Document-level permissions
* OCR and table extraction
* PII detection and protection
* Audit logging
* Background document processing
* Retrieval and generation evaluation datasets
* Monitoring and observability
* Model and version management
* Retry and failure handling
* Production-grade security

---

# 💡 Project Highlights

DocIntel AI demonstrates an end-to-end **LLM engineering and Document AI pipeline**:

```text
Document Processing
        │
        ▼
    Embeddings
        │
        ▼
 Vector Search + BM25
        │
        ▼
 Hybrid Retrieval
        │
        ▼
    Reranking
        │
        ▼
       RAG
        │
        ▼
   Local LLM
        │
        ▼
Structured Extraction
        │
        ▼
Pydantic Validation
        │
        ▼
 JSON / Excel Export
```

The project showcases practical implementation of:

* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Semantic search
* Hybrid retrieval
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Natural Language Processing
* Information extraction
* Vector databases
* Structured generation
* Pydantic validation
* Local AI inference
* Document intelligence

---

# 👤 Author

**Naeim Mousavi**

DocIntel AI — **Enterprise Document Intelligence Engine**

© 2026 Naeim Mousavi

---

# 📄 License

This project is licensed under the **MIT License**.

You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the conditions of the MIT License.

See the `LICENSE` file for the complete license text.
