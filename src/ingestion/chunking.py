"""
Semantic & Table-Preserving Chunking for Financial Documents.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional


class FinancialChunker:
    """Splits markdown pages into chunks while preserving table integrity, column headers, and metadata."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def contains_table(self, text: str) -> bool:
        """
        Robust check to determine if a text block contains markdown table rows.
        Handles possible list prefixes like '- | Col |' or multiple table lines.
        """
        # Look for standard markdown table separator: |---|---| or - |---|---|
        if re.search(r"(?:^|\n)\s*(?:-\s*)?\|(?:\s*[-:]+\s*\|)+", text):
            return True
        # Or look for multiple consecutive lines with table pipe columns
        pipe_lines = [l for l in text.split("\n") if l.count("|") >= 2]
        return len(pipe_lines) >= 2

    @staticmethod
    def extract_table_header(text: str) -> Optional[str]:
        """Extracts the header and separator lines of a markdown table if present."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for i in range(len(lines) - 1):
            if lines[i].count("|") >= 2 and re.search(r"(?:^|\n)\s*(?:-\s*)?\|(?:\s*[-:]+\s*\|)+", lines[i + 1]):
                # Normalize line prefix if preceded by bullet
                header = re.sub(r"^-\s*", "", lines[i])
                sep = re.sub(r"^-\s*", "", lines[i + 1])
                return f"{header}\n{sep}"
        return None

    def split_page(self, page_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Splits a single page markdown content into manageable chunks with overlap.
        Guarantees that tables broken across chunks retain their column headers.
        """
        content = page_data["content"]
        doc_name = page_data["document_name"]
        page_num = page_data["page_number"]

        has_tbl = self.contains_table(content)
        table_header = self.extract_table_header(content) if has_tbl else None

        # If content fits in a single chunk, return intact
        if len(content) <= self.chunk_size:
            return [{
                "chunk_id": f"{doc_name}_p{page_num}_c0",
                "document_name": doc_name,
                "page_number": page_num,
                "content": content,
                "has_table": has_tbl,
            }]

        chunks = []
        paragraphs = re.split(r"\n\s*\n", content)
        current_chunk = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size and we already have content
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": f"{doc_name}_p{page_num}_c{chunk_idx}",
                    "document_name": doc_name,
                    "page_number": page_num,
                    "content": current_chunk.strip(),
                    "has_table": self.contains_table(current_chunk),
                })
                chunk_idx += 1

                # If subsequent paragraph is part of a table and we have a table header, prepend it
                if self.contains_table(para) and table_header and table_header not in para:
                    current_chunk = f"{table_header}\n{para}"
                elif len(current_chunk) > self.chunk_overlap and not self.contains_table(current_chunk):
                    current_chunk = current_chunk[-self.chunk_overlap:] + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

        if current_chunk.strip():
            chunks.append({
                "chunk_id": f"{doc_name}_p{page_num}_c{chunk_idx}",
                "document_name": doc_name,
                "page_number": page_num,
                "content": current_chunk.strip(),
                "has_table": self.contains_table(current_chunk),
            })

        return chunks

    def process_and_save(self, pages: List[Dict[str, Any]], output_path: Path) -> List[Dict[str, Any]]:
        """Chunks all parsed pages and exports to a JSONL file."""
        all_chunks = []
        for p in pages:
            all_chunks.extend(self.split_page(p))

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ch in all_chunks:
                f.write(json.dumps(ch, ensure_ascii=False) + "\n")

        return all_chunks
