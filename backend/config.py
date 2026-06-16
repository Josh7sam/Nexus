"""
Central configuration module for Nexus.
Loads values from .env and exposes typed settings using Pydantic Settings.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Fall back to environment variable values if local .env is missing
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-2"
    
    # ChromaDB (Dense Retrieval)
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "Nexus_dense"
    
    # BM25 (Sparse Retrieval)
    BM25_INDEX_PATH: str = "./data/bm25_index.pkl"
    BM25_CORPUS_PATH: str = "./data/bm25_corpus.pkl"
    
    # Retrieval Parameters
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    FUSION_TOP_K: int = 3
    RRF_K_CONSTANT: int = 60
    DENSE_WEIGHT: float = 0.6
    SPARSE_WEIGHT: float = 0.4
    
    # Ingestion
    CHUNK_SIZE: int = 450
    CHUNK_OVERLAP: int = 45
    DATA_DIR: str = "./data/documents"
    
    # Agent
    MAX_REWRITE_ATTEMPTS: int = 3
    MAX_HALLUCINATION_RETRIES: int = 0
    
    # Feedback / RLHF
    FEEDBACK_DB_PATH: str = "./data/feedback.db"
    
    # Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

# Resolve API Keys: prefer GOOGLE_API_KEY, fallback to GEMINI_API_KEY
GOOGLE_API_KEY = settings.GOOGLE_API_KEY or settings.GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")
GEMINI_API_KEY = settings.GEMINI_API_KEY or settings.GOOGLE_API_KEY or os.getenv("GEMINI_API_KEY", "") or os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = settings.GEMINI_MODEL
GEMINI_EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL

# Expose fields for module-level compatibility
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def make_absolute(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT_DIR, path))

CHROMA_PERSIST_DIR = make_absolute(settings.CHROMA_PERSIST_DIR)
CHROMA_COLLECTION_NAME = settings.CHROMA_COLLECTION_NAME
BM25_INDEX_PATH = make_absolute(settings.BM25_INDEX_PATH)
BM25_CORPUS_PATH = make_absolute(settings.BM25_CORPUS_PATH)
DENSE_TOP_K = settings.DENSE_TOP_K
SPARSE_TOP_K = settings.SPARSE_TOP_K
FUSION_TOP_K = settings.FUSION_TOP_K
RRF_K_CONSTANT = settings.RRF_K_CONSTANT
DENSE_WEIGHT = settings.DENSE_WEIGHT
SPARSE_WEIGHT = settings.SPARSE_WEIGHT
CHUNK_SIZE = settings.CHUNK_SIZE
CHUNK_OVERLAP = settings.CHUNK_OVERLAP
DATA_DIR = make_absolute(settings.DATA_DIR)
MAX_REWRITE_ATTEMPTS = settings.MAX_REWRITE_ATTEMPTS
MAX_HALLUCINATION_RETRIES = settings.MAX_HALLUCINATION_RETRIES
FEEDBACK_DB_PATH = make_absolute(settings.FEEDBACK_DB_PATH)
API_HOST = settings.API_HOST
API_PORT = settings.API_PORT
DEBUG = settings.DEBUG
