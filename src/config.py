"""Configuration settings using Pydantic."""

import os
import tempfile
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    # Provider & Model Options
    DEFAULT_LLM_PROVIDER: str = "openai"  # "openai" or "gemini"
    DEFAULT_OPENAI_MODEL: str = "gpt-4o-mini"
    DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
    DEFAULT_LLM_MODEL: str = "gpt-4o-mini"
    DEFAULT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Paths
    BASE_DIR: Path = PROJECT_ROOT
    DATA_RAW_DIR: Path = PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"
    VECTOR_STORE_PATH: Path = PROJECT_ROOT / "data" / "chroma_db"
    FLASHRANK_CACHE_DIR: Path = Path(tempfile.gettempdir()) / "flashrank"
    
    # Collection & Retrieval
    CHROMA_COLLECTION_NAME: str = "financial_reports_minilm"
    TOP_K_RETRIEVAL: int = 8
    TOP_K_RERANK: int = 4
    RERANKER_MODEL: str = "flashrank"

    # Server & Security Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501", "http://127.0.0.1:8501"]
    MAX_UPLOAD_SIZE_BYTES: int = 25 * 1024 * 1024  # 25 MB

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
