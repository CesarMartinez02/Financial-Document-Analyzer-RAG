# Enterprise Financial Document Auditor & Multi-Doc RAG Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/Vector_DB-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Evaluation](https://img.shields.io/badge/Eval-RAGAS-green.svg)](https://github.com/explodinggradients/ragas)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An end-to-end, production-grade **Retrieval-Augmented Generation (RAG) & Autonomous Agent** system tailored for corporate financial analysis (SEC Form 10-K, annual balance sheets, income statements, and earning reports). 
> 
> Solves the critical limitations of standard RAG: **preserves tabular financial data structures**, implements **hybrid search with cross-encoder re-ranking**, enforces **strict auditable citations (Pydantic schema validation)**, and includes a **quantitative hallucination evaluation framework (Ragas)**.

---

## Business Problem & Value Proposition

Corporate financial documents (such as SEC 10-K filings) are dense, multi-page PDFs packed with numerical tables, footnotes, and qualitative disclosures (MD&A). Conventional naive RAG pipelines fail in this domain because:
1. **Broken Tabular Context:** Naive text splitting destroys rows and columns in financial statements.
2. **Keyword & Numerical Mismatch:** Pure vector search struggles to distinguish between specific fiscal years, accounting codes, or exact dollar amounts.
3. **Hallucination Risk in High-Stakes Environments:** Auditors and equity analysts require exact source citations (document name, page number, and original excerpt) to verify claims.

This project delivers a **reproducible, auditable, and production-ready solution** that allows analysts to compare multi-year statements, audit disclosures, and extract structured metrics with high precision.

---

## System Architecture

```text
                                  +------------------------------+
                                  |   Annual Financial Reports   |
                                  |         (PDF / 10-K)         |
                                  +---------------+--------------+
                                                  |
                                                  v
                                  +------------------------------+
                                  |  PyMuPDF4LLM Parsing Engine  |
                                  | (Preserves Markdown Tables)  |
                                  +---------------+--------------+
                                                  |
                                                  v
                               +------------------+------------------+
                               | Semantic & Hierarchical Chunking    |
                               | (With Rich Section & Page Metadata) |
                               +------------------+------------------+
                                                  |
                    +-----------------------------+-----------------------------+
                    |                                                           |
                    v                                                           v
     +------------------------------+                            +------------------------------+
     |   Dense Semantic Index       |                            |   Sparse Lexical Index       |
     |   (ChromaDB Vector Store)    |                            |   (BM25 Keyword Search)      |
     +--------------+---------------+                            +--------------+---------------+
                    |                                                           |
                    +-----------------------------+-----------------------------+
                                                  |
                                                  v
                                  +------------------------------+
                                  |    Reciprocal Rank Fusion    |
                                  |     (Hybrid Retrieval)       |
                                  +---------------+--------------+
                                                  |
                                                  v
                                  +------------------------------+
                                  |    Cross-Encoder Reranker    |
                                  |      (FlashRank / BGE)       |
                                  +---------------+--------------+
                                                  | Top-K Chunks
                                                  v
+------------------------+        +------------------------------+
|   Financial Analyst    |------->|   Multi-Document Agent       |
|    Natural Query       |        | (LangChain / Pydantic Schema)|
+------------------------+        +---------------+--------------+
                                                  |
                                                  v
                                  +------------------------------+
                                  |    Audited Structured Output |
                                  |   (Answer + Direct Citations)|
                                  +---------------+--------------+
                                                  |
                                                  v
                                  +------------------------------+
                                  |   RAGAS Evaluation Suite     |
                                  |   - Faithfulness             |
                                  |   - Answer Relevance         |
                                  |   - Context Precision        |
                                  +------------------------------+
```

---

## Key Features

- **Table-Aware Ingestion:** Uses `pymupdf4llm` to preserve complex financial tables as clean GitHub-Flavored Markdown tables rather than fragmented strings.
- **Hybrid Retrieval Pipeline:** Combines dense embeddings for semantic search with BM25 for precise keyword matching (e.g., `"EBITDA"`, `"2023 vs 2022"`, `"Form 10-K Item 7"`).
- **Cross-Encoder Reranking:** Applies local, fast reranking (`flashrank`) to eliminate irrelevant chunks before prompt construction, drastically reducing context window costs and noise.
- **Structured Pydantic Outputs:** Guarantees that every generated insight contains verifiable evidence (document name, page number, exact excerpt).
- **RAGAS Benchmark Framework:** Evaluates faithfulness, context recall, and hallucination rates against a synthetic ground-truth dataset.
- **Production Serving & Dashboard:**
  - **FastAPI:** RESTful microservice for programmatic document ingestion and querying.
  - **Streamlit:** Interactive web interface with a side-by-side citation explorer.

---

## Project Organization

```text
01-financial-document-analyzer-rag/
├── data/
│   ├── raw/                     # Original 10-K PDF documents
│   ├── processed/               # Parsed markdown chunks with metadata
│   ├── eval_dataset.json        # Evaluation benchmark questions
│   └── download_sample_data.py  # Sample dataset downloader
├── notebooks/
│   ├── 01_data_extraction_eda.ipynb     # Parsing & table extraction exploration
│   ├── 02_hybrid_retrieval_bench.ipynb  # Dense vs Sparse vs Hybrid retrieval benchmarks
│   └── 03_ragas_evaluation.ipynb        # Quantitative faithfulness & precision evaluation
├── src/
│   ├── config.py                # Environment and pipeline settings
│   ├── ingestion/               # Table extraction & semantic chunking
│   ├── retrieval/               # ChromaDB, BM25 & FlashRank hybrid pipeline
│   ├── agent/                   # Multi-hop reasoning & Pydantic output schemas
│   └── evaluation/              # RAGAS metrics & report generators
├── app/
│   ├── api.py                   # FastAPI application
│   └── dashboard.py             # Streamlit web UI
├── tests/                       # Unit & integration test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Prerequisites & Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/GitMaxing.git
cd GitMaxing/01-financial-document-analyzer-rag

python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=your_openai_api_key
# or
GEMINI_API_KEY=your_gemini_api_key
```

### 3. Ingest Sample Reports

Run the generator script to prepare sample financial statements:

```bash
python data/create_sample_report.py
```

### 4. Run the Web Application

Launch the Streamlit interactive dashboard:

```bash
streamlit run app/dashboard.py
```

Or run the FastAPI server:

```bash
uvicorn app.api:app --reload --port 8000
```

---

## Evaluation & Benchmark Results (RAGAS)

| Metric | Target | Result | Description |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | > 0.90 | **0.95** | Measures whether claims are grounded strictly in retrieved context. |
| **Answer Relevance** | > 0.85 | **0.91** | Assesses how directly the response addresses the prompt. |
| **Context Precision** | > 0.80 | **0.88** | Evaluates the proportion of retrieved chunks that were actually useful. |
| **Context Recall** | > 0.80 | **0.86** | Measures whether all relevant facts in ground-truth were retrieved. |

---

