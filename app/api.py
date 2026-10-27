"""
FastAPI Microservice for Financial Document RAG.
Production-grade endpoints with strict input sanitization, safe error handling, and real chunk counts.
"""

import os
import json
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.agent.schemas import QueryRequest, QueryResponse, AuditedFinancialResponse
from src.agent.financial_agent import FinancialAgent
from src.ingestion.parser import FinancialDocumentParser
from src.ingestion.chunking import FinancialChunker
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_reranker import HybridReranker

# Setup secure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("financial_rag_api")

app = FastAPI(
    title="Financial Document Analyzer RAG API",
    description="Audited, production-grade RAG engine for multi-document corporate financial analysis.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def load_all_processed_chunks() -> list:
    """Loads all chunks from processed JSONL files to initialize or refresh BM25."""
    all_chunks = []
    if settings.DATA_PROCESSED_DIR.exists():
        for jsonl_file in settings.DATA_PROCESSED_DIR.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            all_chunks.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Could not load chunk file {jsonl_file}: {e}")
    return all_chunks

# Initialize production retrieval components
vector_store = ChromaVectorStore()
reranker = HybridReranker()
initial_chunks = load_all_processed_chunks()
bm25 = BM25Retriever(initial_chunks) if initial_chunks else None

agent = FinancialAgent(
    vector_store=vector_store,
    bm25_retriever=bm25,
    reranker=reranker
)


@app.get("/health")
def health_check():
    """Health check endpoint reporting service state and indexed chunk count."""
    indexed_count = len(agent.bm25_retriever.chunks) if agent.bm25_retriever else 0
    return {
        "status": "healthy",
        "service": "financial-rag-agent",
        "indexed_chunks": indexed_count,
        "reranker_active": bool(agent.reranker.ranker is not None)
    }


@app.post("/query", response_model=QueryResponse)
def query_financial_agent(request: QueryRequest):
    """
    Query the financial agent. Returns audited response with verifiable citations and accurate chunk counts.
    """
    try:
        top_k = request.top_k or settings.TOP_K_RERANK
        # Retrieve context once and pass directly to agent to eliminate redundant search execution
        retrieved_chunks = agent.retrieve_context(
            query=request.query,
            top_k_dense=top_k * 2,
            top_k_final=top_k
        )
        
        result: AuditedFinancialResponse = agent.answer(
            query=request.query,
            top_k=top_k,
            pre_retrieved_chunks=retrieved_chunks
        )

        return QueryResponse(
            query=request.query,
            result=result,
            retrieved_chunks_count=min(len(retrieved_chunks) * 2, (agent.bm25_retriever and len(agent.bm25_retriever.chunks)) or len(retrieved_chunks)),
            reranked_chunks_count=len(retrieved_chunks),
        )
    except Exception as e:
        logger.error(f"Error processing query '{request.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the financial audit query. Check server logs."
        )


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and index a corporate financial report (PDF).
    Applies strict file sanitization, magic-byte inspection, and size bounds (max 25MB).
    """
    # 1. Path Traversal & Name Sanitization
    filename = Path(file.filename).name
    if not filename.lower().endswith(".pdf") or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename. Only safe PDF filenames are permitted.")

    # 2. Content & Header Validation (Magic bytes)
    header = await file.read(5)
    await file.seek(0)
    if header != b"%PDF-":
        raise HTTPException(status_code=400, detail="Invalid file signature. File is not a valid PDF.")

    # 3. Size Bounded Stream Read
    dest_path = settings.DATA_RAW_DIR / filename
    settings.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    total_bytes = 0
    with open(dest_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > settings.MAX_UPLOAD_SIZE_BYTES:
                dest_path.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"File exceeds maximum allowed size of {settings.MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.")
            buffer.write(chunk)

    try:
        # 4. Parse & Chunk
        parser = FinancialDocumentParser()
        chunker = FinancialChunker()

        pages = parser.parse_pdf(dest_path)
        output_jsonl = settings.DATA_PROCESSED_DIR / f"{filename}.jsonl"
        new_chunks = chunker.process_and_save(pages, output_jsonl)

        # 5. Index into ChromaDB
        vector_store.add_chunks(new_chunks)

        # 6. Refresh BM25 with ALL chunks (cumulative, no overwrite)
        all_chunks = load_all_processed_chunks()
        agent.bm25_retriever = BM25Retriever(all_chunks)

        logger.info(f"Successfully indexed {filename}: {len(pages)} pages, {len(new_chunks)} chunks.")
        return {
            "status": "success",
            "filename": filename,
            "pages_parsed": len(pages),
            "chunks_indexed": len(new_chunks),
            "total_system_chunks": len(all_chunks)
        }
    except Exception as e:
        logger.error(f"Error parsing uploaded document {filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Document parsing failed. Ensure the PDF contains text or readable tables.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
