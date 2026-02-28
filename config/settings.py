import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str
    firecrawl_api_key: str
    embeddings_model: str = "AITeamVN/Vietnamese_Embedding_v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    # gemini-2.5-flash
    llm_model: str = "gemma-3-27b-it"

    vector_dim: int = 1024

    top_k: int = 3
    batch_size: int = 16
    chunk_size: int = 2048
    chunk_overlap: int = 256
    re_rank_top_k: int = 3

    qdrant_db_path: str = "./data/qdrant_db"
    collection_name: str = "legal_agent"
    qdrant_api_key: str
    qdrant_url: str
    docs_path: str = "./data/corpus.json"

    hf_cache_dir: str = "./cache/hf_cache"

    temperature: float = 0.7
    max_tokens: int = 2048

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
