"""
Dense Vector Store management using ChromaDB.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
from chromadb.utils import embedding_functions
from src.config import settings


class ChromaVectorStore:
    """Manages persistent ChromaDB vector collection for financial documents."""

    def __init__(self, persist_dir: Path = settings.VECTOR_STORE_PATH, collection_name: str = None):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Consistent, local-first embedding function to guarantee dimensional stability (384 dims)
        # Allows OpenAI/Gemini to focus on generation without mixing vector spaces.
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        final_collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        
        self.collection = self.client.get_or_create_collection(
            name=final_collection_name,
            embedding_function=self.embedding_fn
        )

    def _get_embedding_function(self):
        """Initializes appropriate embedding function based on config."""
        if settings.OPENAI_API_KEY:
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.OPENAI_API_KEY,
                model_name=settings.DEFAULT_EMBEDDING_MODEL
            )
        elif settings.GEMINI_API_KEY:
            return embedding_functions.GoogleGenerativeAiEmbeddingFunction(
                api_key=settings.GEMINI_API_KEY,
                model_name="models/embedding-001"
            )
        else:
            # Default to Chroma's built-in all-MiniLM-L6-v2 ONNX local embedding
            return embedding_functions.DefaultEmbeddingFunction()

    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Indexes processed chunks into ChromaDB."""
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "document_name": c["document_name"],
                "page_number": int(c["page_number"]),
                "has_table": bool(c.get("has_table", False)),
            }
            for c in chunks
        ]

        # Chroma handles batching automatically or in blocks
        batch_size = 200
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                ids=ids[i:i + batch_size],
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
            )

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Queries ChromaDB for semantic similarity.
        Returns list of (chunk_dict, distance_score).
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        scored_chunks = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            ids = results["ids"][0]
            distances = results["distances"][0]

            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances):
                chunk_data = {
                    "chunk_id": doc_id,
                    "content": doc,
                    "document_name": meta.get("document_name", "unknown"),
                    "page_number": meta.get("page_number", 1),
                    "has_table": meta.get("has_table", False),
                }
                # Chroma distance is L2/cosine distance; convert to similarity score
                similarity = 1.0 / (1.0 + float(dist))
                scored_chunks.append((chunk_data, similarity))

        return scored_chunks
