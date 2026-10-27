"""
PDF Parsing Engine specialized for financial documents.
Preserves complex balance sheets, income statements, and cash flows in clean Markdown tables.
"""

from pathlib import Path
from typing import List, Dict, Any
import pymupdf as fitz

try:
    import pymupdf4llm
    PYMUPDF4LLM_AVAILABLE = True
except ImportError:
    PYMUPDF4LLM_AVAILABLE = False


class FinancialDocumentParser:
    """Extracts structured text and markdown tables from PDF filings."""

    def __init__(self):
        pass

    def parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parses a PDF file page-by-page, converting tables to markdown format.

        Returns:
            List of dicts with:
                - document_name: str
                - page_number: int (1-indexed)
                - content: str (markdown format)
                - total_pages: int
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        document_name = file_path.name
        pages_data = []

        if PYMUPDF4LLM_AVAILABLE:
            # Use PyMuPDF4LLM for advanced table-to-markdown extraction
            try:
                # pymupdf4llm can output page-by-page markdown chunks
                md_pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
                total_pages = len(md_pages)
                for idx, item in enumerate(md_pages):
                    page_num = item.get("metadata", {}).get("page_number", idx + 1)
                    content = item.get("text", "").strip()
                    if content:
                        pages_data.append({
                            "document_name": document_name,
                            "page_number": page_num,
                            "content": content,
                            "total_pages": total_pages,
                        })
                return pages_data
            except Exception as e:
                print(f"[Warning] pymupdf4llm extraction failed: {e}. Falling back to standard PyMuPDF.")

        # Fallback to standard PyMuPDF page-by-page text extraction
        doc = fitz.open(file_path)
        total_pages = len(doc)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if text:
                pages_data.append({
                    "document_name": document_name,
                    "page_number": page_idx + 1,
                    "content": text,
                    "total_pages": total_pages,
                })
        doc.close()

        return pages_data
