"""
BM25 Keyword Retriever for exact financial terms, codes, and numerical figures.
"""

import re
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """In-memory BM25 index over financial chunks."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks
        self.corpus = [self._tokenize(c["content"]) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus) if self.corpus else None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple lowercase alphanumeric tokenization."""
        return re.findall(r"\w+", text.lower())

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches the BM25 index and returns top-k chunks with their BM25 score.
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Pair chunk with score and sort descending
        scored_chunks = list(zip(self.chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        return scored_chunks[:top_k]
