"""
Hybrid Retrieval with Reciprocal Rank Fusion (RRF) and Cross-Encoder Re-ranking.
"""

import tempfile
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from pathlib import Path
from src.config import settings

try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False


class HybridReranker:
    """Combines BM25 and Dense Retrieval via RRF, then applies Cross-Encoder re-ranking."""

    def __init__(self, rrf_k: int = 60, reranker_model: str = "ms-marco-TinyBERT-L-2-v2"):
        self.rrf_k = rrf_k
        self.ranker = None
        if FLASHRANK_AVAILABLE:
            try:
                # Find available local cache or use temp dir
                candidate_dirs = [
                    Path("C:/tmp/flashrank"),
                    settings.FLASHRANK_CACHE_DIR,
                    Path(tempfile.gettempdir()) / "flashrank",
                ]
                cache_path = next((d for d in candidate_dirs if d.exists()), settings.FLASHRANK_CACHE_DIR)
                cache_path.mkdir(parents=True, exist_ok=True)
                self.ranker = Ranker(model_name=reranker_model, cache_dir=str(cache_path))
            except Exception as e:
                print(f"[Warning] FlashRank initialization failed: {e}. Falling back to RRF.")

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Tuple[Dict[str, Any], float]],
        sparse_results: List[Tuple[Dict[str, Any], float]],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Fuses ranked lists from dense vector search and sparse BM25 using RRF.
        Formula: RRF_score(d) = SUM(1 / (k + rank(d)))
        """
        rrf_scores = defaultdict(float)
        chunk_map = {}

        # Process dense ranking
        for rank, (chunk, _) in enumerate(dense_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank + 1)

        # Process sparse (BM25) ranking
        for rank, (chunk, _) in enumerate(sparse_results):
            cid = chunk["chunk_id"]
            chunk_map[cid] = chunk
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by composite RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        return [chunk_map[cid] for cid in sorted_ids[:top_k]]

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Applies cross-encoder re-ranking to top candidate chunks.
        """
        if not candidates:
            return []

        if self.ranker and FLASHRANK_AVAILABLE:
            try:
                passages = [
                    {"id": c["chunk_id"], "text": c["content"], "meta": c}
                    for c in candidates
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                ranked_results = self.ranker.rerank(rerank_request)
                
                # Extract top_k chunks in reranked order
                reranked_chunks = []
                for res in ranked_results[:top_k]:
                    chunk_data = res["meta"].copy()
                    chunk_data["rerank_score"] = float(res.get("score", 0.0))
                    reranked_chunks.append(chunk_data)
                return reranked_chunks
            except Exception as e:
                print(f"[Warning] Reranker error: {e}. Returning unranked candidates.")

        # Fallback: return top_k candidates directly from RRF
        return candidates[:top_k]
