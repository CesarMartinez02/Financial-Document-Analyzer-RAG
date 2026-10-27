"""
Streamlit Web Dashboard for Financial Document Analyzer & Auditor.
Production-ready interface with isolated session-state security, no emojis, and safe upload validation.
"""

import os
import json
from pathlib import Path
import streamlit as st

from src.config import settings
from src.agent.financial_agent import FinancialAgent
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.hybrid_reranker import HybridReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.ingestion.parser import FinancialDocumentParser
from src.ingestion.chunking import FinancialChunker

# Professional page configuration without decorative emojis
st.set_page_config(
    page_title="Financial Document Auditor & RAG Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Financial Document Auditor & Multi-Document RAG")
st.markdown(
    "Automated retrieval and analysis of corporate financial reports (SEC Form 10-K, annual balance sheets, and earnings statements) with auditable citations."
)

@st.cache_resource
def get_shared_retrieval_backend():
    """Caches heavy retrieval indexes (ChromaDB, Reranker, BM25) across sessions without storing user API keys."""
    vector_store = ChromaVectorStore()
    reranker = HybridReranker()

    chunks = []
    if settings.DATA_PROCESSED_DIR.exists():
        for jsonl_file in settings.DATA_PROCESSED_DIR.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            chunks.append(json.loads(line))
            except Exception:
                pass

    bm25 = BM25Retriever(chunks) if chunks else None
    return vector_store, reranker, bm25

vector_store, reranker, bm25 = get_shared_retrieval_backend()

# Security: Isolate Agent instance and user API keys per session state
if "agent" not in st.session_state:
    st.session_state.agent = FinancialAgent(
        vector_store=vector_store,
        bm25_retriever=bm25,
        reranker=reranker
    )

agent: FinancialAgent = st.session_state.agent
# Sync shared BM25 if new docs were indexed
agent.bm25_retriever = bm25
agent.vector_store = vector_store

# Sidebar: Configuration and Document Management
with st.sidebar:
    st.header("LLM Provider Configuration")
    provider_choice = st.selectbox(
        "Select Provider:",
        options=["Google Gemini (Recommended)", "OpenAI", "None (Local Extractive Only)"],
        index=0
    )

    if provider_choice == "Google Gemini (Recommended)":
        user_key = st.text_input("Gemini API Key:", type="password", help="Enter your Gemini API key from Google AI Studio")
        if user_key:
            agent.set_credentials(user_key, provider="gemini")
            st.success("Gemini API key configured for current session.")
        else:
            agent.llm = None
    elif provider_choice == "OpenAI":
        user_key = st.text_input("OpenAI API Key:", type="password", help="Enter your OpenAI API key")
        if user_key:
            agent.set_credentials(user_key, provider="openai")
            st.success("OpenAI API key configured for current session.")
        else:
            agent.llm = None
    else:
        agent.llm = None
        st.info("Running in local deterministic extractive mode. No API key needed.")

    st.divider()
    st.header("Document Management")
    uploaded_file = st.file_uploader("Upload Corporate Filing (PDF):", type=["pdf"])

    if uploaded_file and st.button("Index Document", type="primary"):
        # Security: sanitize filename and validate PDF
        safe_filename = Path(uploaded_file.name).name
        file_bytes = uploaded_file.getbuffer()

        if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
            st.error(f"File exceeds maximum size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.")
        elif not file_bytes[:5].tobytes() == b"%PDF-":
            st.error("Uploaded file signature is not a valid PDF.")
        else:
            with st.spinner(f"Parsing and indexing {safe_filename}..."):
                dest_path = settings.DATA_RAW_DIR / safe_filename
                settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
                with open(dest_path, "wb") as f:
                    f.write(file_bytes)

                parser = FinancialDocumentParser()
                chunker = FinancialChunker()
                pages = parser.parse_pdf(dest_path)

                output_jsonl = settings.DATA_PROCESSED_DIR / f"{safe_filename}.jsonl"
                chunks = chunker.process_and_save(pages, output_jsonl)

                vector_store.add_chunks(chunks)

                # Clear cache and refresh BM25
                st.cache_resource.clear()
                st.success(f"Indexed {len(pages)} pages into {len(chunks)} searchable chunks.")
                st.rerun()

    st.divider()
    st.subheader("System Status")
    st.write(f"Embedding Model: `all-MiniLM-L6-v2` (Local, 384 dims)")
    st.write(f"Reranker: `FlashRank / RRF`")
    st.write(f"Indexed System Chunks: `{len(bm25.chunks) if bm25 else 0}`")
    st.write(f"Active Mode: `{'Generative (' + agent.provider.title() + ')' if agent.llm else 'Local Extractive'}`")

# Main Interface: Query
sample_queries = [
    "What was the Net Income from 2021 to 2023?",
    "Compare Operating Income in 2021 vs 2022",
    "What were Apple sales in numbers during fiscal 2023?",
    "What are the principal risk factors regarding supply chain?",
]

selected_sample = st.selectbox("Quick Query Presets:", ["Type custom query..."] + sample_queries)
default_text = selected_sample if selected_sample != "Type custom query..." else ""

query_text = st.text_area("Analytical Question:", value=default_text, height=90)
top_k_val = st.slider("Context chunks to retain after reranking:", min_value=2, max_value=8, value=4)

if st.button("Run Financial Audit", type="primary"):
    if not query_text.strip():
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("Executing hybrid retrieval, reranking, and analytical breakdown..."):
            response = agent.answer(query=query_text, top_k=top_k_val)

        # 1. Executive Summary
        st.subheader("Executive Summary")
        st.info(response.executive_summary)

        # 2. Extracted Metrics
        if response.key_metrics:
            st.subheader("Key Extracted Figures")
            cols = st.columns(min(len(response.key_metrics), 4))
            for i, metric in enumerate(response.key_metrics):
                col = cols[i % len(cols)]
                with col:
                    st.metric(
                        label=f"{metric.metric_name}",
                        value=metric.value,
                        help=f"Company: {metric.company} | Period: {metric.fiscal_period}",
                    )

        # 3. Detailed Breakdown
        st.subheader("Detailed Analysis & Evidence")
        st.markdown(response.detailed_analysis)

        # 4. Confidence Score
        st.markdown(f"**Confidence Score:** `{response.confidence_score * 100:.1f}%`")
        st.progress(response.confidence_score)

        # 5. Audited Citations
        st.subheader("Audited Source Citations")
        if response.citations:
            for idx, cit in enumerate(response.citations):
                with st.expander(f"Citation [{idx + 1}] | {cit.document_name} - Page {cit.page_number}"):
                    st.markdown(f"**Quote:** *\"{cit.exact_quote}\"*")
                    if cit.relevance_explanation:
                        st.caption(f"Relevance: {cit.relevance_explanation}")
        else:
            st.caption("No direct quotations found.")
