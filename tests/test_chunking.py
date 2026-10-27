"""Unit tests for table detection and chunking."""

from src.ingestion.chunking import FinancialChunker


def test_table_detection():
    chunker = FinancialChunker()
    markdown_table = """
| Metric | 2022 | 2023 |
|---|---|---|
| Total Revenue | $394,328 | $383,285 |
| Gross Margin | 43.3% | 44.1% |
"""
    plain_text = "This is a regular narrative paragraph discussing supply chain constraints."
    
    assert chunker.contains_table(markdown_table) is True
    assert chunker.contains_table(plain_text) is False


def test_split_page_preserves_metadata():
    chunker = FinancialChunker(chunk_size=100, chunk_overlap=20)
    page_data = {
        "document_name": "sample_10k.pdf",
        "page_number": 5,
        "content": "Paragraph 1 is here.\n\nParagraph 2 is slightly longer to cause a split in chunking.\n\nParagraph 3 is the final block.",
    }
    
    chunks = chunker.split_page(page_data)
    assert len(chunks) > 0
    for ch in chunks:
        assert ch["document_name"] == "sample_10k.pdf"
        assert ch["page_number"] == 5
        assert "chunk_id" in ch
