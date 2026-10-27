"""
End-to-End verification test: Parsing -> Chunking -> ChromaDB -> BM25 -> Hybrid Retrieval.
"""

from pathlib import Path
from src.ingestion.parser import FinancialDocumentParser
from src.ingestion.chunking import FinancialChunker
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_reranker import HybridReranker
from src.agent.financial_agent import FinancialAgent

def test_full_pipeline():
    sample_pdf = Path(__file__).resolve().parent.parent / "data" / "raw" / "apple_10k_sample.pdf"
    assert sample_pdf.exists(), "Sample PDF must exist"

    # 1. Parse
    parser = FinancialDocumentParser()
    pages = parser.parse_pdf(sample_pdf)
    assert len(pages) == 4, f"Expected 4 pages, got {len(pages)}"

    # 2. Chunk
    chunker = FinancialChunker(chunk_size=1200, chunk_overlap=150)
    output_jsonl = Path(__file__).resolve().parent.parent / "data" / "processed" / "apple_10k_sample.jsonl"
    chunks = chunker.process_and_save(pages, output_jsonl)
    assert len(chunks) >= 4, "Expected at least 4 chunks"

    # 3. Dense Index (ChromaDB)
    vector_store = ChromaVectorStore()
    vector_store.add_chunks(chunks)

    # 4. Sparse Index (BM25)
    bm25 = BM25Retriever(chunks)

    # 5. Hybrid Retrieval & Rerank
    reranker = HybridReranker()
    agent = FinancialAgent(vector_store=vector_store, bm25_retriever=bm25, reranker=reranker)

    # Query for financial metric
    query = "What was the total net sales in 2023?"
    retrieved = agent.retrieve_context(query, top_k_dense=6, top_k_final=4)

    assert len(retrieved) > 0
    # Context should include financial figures from operations statement or MD&A
    found_number = any("383,285" in ch["content"] or "net sales" in ch["content"].lower() for ch in retrieved)
    assert found_number, "Top retrieved chunks must include net sales figures"

    # 6. Test Agent Structured Answer (Mock / Fallback mode when no API key is set)
    answer = agent.answer(query, top_k=4)
    assert answer.confidence_score >= 0.0
    assert len(answer.citations) > 0
    print("\n[SUCCESS] Pipeline executed cleanly!")
    print(f"Summary: {answer.executive_summary}")
    print(f"Top citation page: {answer.citations[0].page_number}")
