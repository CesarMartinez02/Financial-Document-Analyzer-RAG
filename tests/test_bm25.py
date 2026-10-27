"""Unit tests for BM25 retrieval."""

from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_search_scoring():
    sample_chunks = [
        {"chunk_id": "c1", "content": "Apple reported record Services revenue in fiscal 2023.", "page_number": 1, "document_name": "apple.pdf"},
        {"chunk_id": "c2", "content": "Microsoft Cloud revenue grew significantly driven by Azure.", "page_number": 2, "document_name": "msft.pdf"},
        {"chunk_id": "c3", "content": "Operating expenses included research and development expenditures.", "page_number": 3, "document_name": "apple.pdf"},
    ]
    
    retriever = BM25Retriever(sample_chunks)
    results = retriever.search("Apple Services revenue", top_k=2)

    assert len(results) > 0
    top_chunk, score = results[0]
    assert top_chunk["chunk_id"] == "c1"
    assert score > 0
