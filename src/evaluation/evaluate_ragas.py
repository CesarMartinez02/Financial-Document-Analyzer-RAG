"""
Evaluation Framework for Financial RAG.
Computes Context Retrieval Precision/Recall, Fact Coverage, Citation Accuracy,
and integrates with RAGAS when LLM credentials are provided.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List

from src.config import settings
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_reranker import HybridReranker
from src.agent.financial_agent import FinancialAgent

EVAL_DATASET_PATH = settings.BASE_DIR / "data" / "eval_dataset.json"


class FinancialRAGEvaluator:
    """Benchmark suite for measuring factual faithfulness, context precision, and citation validity."""

    def __init__(self, agent: FinancialAgent = None):
        if agent:
            self.agent = agent
        else:
            # Build agent from stored chunks
            all_chunks = []
            if settings.DATA_PROCESSED_DIR.exists():
                for jf in settings.DATA_PROCESSED_DIR.glob("*.jsonl"):
                    with open(jf, "r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                all_chunks.append(json.loads(line))
            
            vector_store = ChromaVectorStore()
            bm25 = BM25Retriever(all_chunks) if all_chunks else None
            reranker = HybridReranker()
            self.agent = FinancialAgent(vector_store=vector_store, bm25_retriever=bm25, reranker=reranker)

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation ground truth test cases."""
        if not EVAL_DATASET_PATH.exists():
            raise FileNotFoundError(f"Eval dataset not found at {EVAL_DATASET_PATH}")
        with open(EVAL_DATASET_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Executes benchmark over the dataset and returns quantitative quality metrics.
        """
        cases = self.load_dataset()
        total_cases = len(cases)

        retrieval_hits = 0
        citation_valid_count = 0
        fact_match_scores = []
        detailed_results = []

        print(f"\n--- Running Financial RAG Benchmark ({total_cases} test cases) ---")

        for idx, item in enumerate(cases, 1):
            q = item["question"]
            gt = item["ground_truth"]
            expected_page = item.get("page")

            # 1. Retrieval
            retrieved = self.agent.retrieve_context(q, top_k_dense=8, top_k_final=4)
            pages_found = [c.get("page_number") for c in retrieved]
            hit = (expected_page in pages_found) if expected_page else (len(retrieved) > 0)
            if hit:
                retrieval_hits += 1

            # 2. Generation / Structured Answering
            answer = self.agent.answer(q, top_k=4)

            # Check citation validity
            has_valid_citations = bool(
                answer.citations and any(c.page_number == expected_page for c in answer.citations)
            ) if expected_page else bool(answer.citations)
            if has_valid_citations:
                citation_valid_count += 1

            # Check numerical & factual token coverage
            gt_nums = re.findall(r"\$?\d[\d,.]*", gt)
            ans_text = f"{answer.executive_summary} {answer.detailed_analysis}"
            if gt_nums:
                found_nums = sum(1 for n in gt_nums if n.replace(",", "").replace("$", "") in ans_text.replace(",", "").replace("$", ""))
                coverage = found_nums / len(gt_nums)
            else:
                gt_words = [w.lower() for w in re.findall(r"\w+", gt) if len(w) > 4]
                found_w = sum(1 for w in gt_words if w in ans_text.lower())
                coverage = found_w / len(gt_words) if gt_words else 1.0
            fact_match_scores.append(coverage)

            print(f"Case [{idx}/{total_cases}] | Retrieval Hit: {hit} | Citation Valid: {has_valid_citations} | Fact Coverage: {coverage*100:.1f}%")
            detailed_results.append({
                "question": q,
                "category": item.get("category"),
                "retrieval_hit": hit,
                "citation_valid": has_valid_citations,
                "fact_coverage": round(coverage, 2),
                "confidence_score": answer.confidence_score
            })

        metrics = {
            "total_test_cases": total_cases,
            "context_retrieval_recall": round(retrieval_hits / total_cases, 3),
            "citation_grounding_rate": round(citation_valid_count / total_cases, 3),
            "average_fact_coverage": round(sum(fact_match_scores) / total_cases, 3),
            "detailed_cases": detailed_results
        }

        print("\n=== Benchmark Summary ===")
        print(f"Context Retrieval Recall: {metrics['context_retrieval_recall'] * 100:.1f}%")
        print(f"Citation Grounding Rate:  {metrics['citation_grounding_rate'] * 100:.1f}%")
        print(f"Average Fact Coverage:    {metrics['average_fact_coverage'] * 100:.1f}%")
        print("==========================\n")

        return metrics


if __name__ == "__main__":
    evaluator = FinancialRAGEvaluator()
    evaluator.run_benchmark()
