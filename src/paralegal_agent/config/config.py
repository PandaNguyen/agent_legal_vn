import os
from pathlib import Path
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    gemini_api_key: str
    huggingface_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("HUGGINGFACE_API_KEY", "HF_TOKEN"),
    )
    firecrawl_api_key: str
    embeddings_model: str = "AITeamVN/Vietnamese_Embedding_v2"
    sparse_embedding_model: str = "Qdrant/bm25"
    # gemini-2.5-flash "gemma-3-27b-it" "ollama/phi3.5" Qwen/Qwen3-4B-Instruct-2507:nscale deepseek-ai/DeepSeek-R1-0528:together
    llm_model: str = "huggingface/deepseek-ai/DeepSeek-R1-0528:together"
    ollama_base_url: str = "http://localhost:11434"
    huggingface_router_url: str = "https://router.huggingface.co/v1/chat/completions"

    vector_dim: int = 1024

    top_k: int = 3
    batch_size: int = 16
    chunk_size: int = 2048
    chunk_overlap: int = 256
    re_rank_top_k: int = 3

    qdrant_db_path: str = "./data/qdrant_db" #if using local Qdrant, otherwise not needed
    qdrant_collection_name: str = "legal_documents"
    qdrant_api_key: str
    qdrant_url: str
    docs_path: str = "./data/corpus.json"

    hf_cache_dir: str = "./cache/hf_cache"

    temperature: float = 0.1
    max_tokens: int = 2048

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
