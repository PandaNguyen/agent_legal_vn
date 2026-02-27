import numpy as np
from typing import List
from loguru import logger
from sentence_transformers import SentenceTransformer
from config.settings import settings

def batch_iterate(lst: List, batch_size: int):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

class Embeddata:
    def __init__(self,
                 embed_model_name: str = None,
                 batch_size: int = None,
                 cache_folder: str = None):
        self.embed_model_name = embed_model_name or settings.embeddings_model
        self.batch_size = batch_size or settings.batch_size
        self.cache_folder = cache_folder or settings.hf_cache_dir

        self.embed_model = self._load_embed_model()
        self.embeddings = []
        self.contexts = []

    def _load_embed_model(self):
        logger.info(f"Loading embedding model: {self.embed_model_name}")
        model = SentenceTransformer(self.embed_model_name, cache_folder=self.cache_folder, trust_remote_code=True)
        return model
    
    def generate_embeddings(self, contexts: List[str]):
        embeddings = self.embed_model.encode(
            sentences=contexts,
            batch_size= min(self.batch_size, max(1, len(contexts))),
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings
    def embed(self, contexts: List[str]):
        self.contexts = contexts
        logger.info(f"Generating embeddings for {len(contexts)} contexts")

        for batch in batch_iterate(contexts, self.batch_size):
            batch_embeddings = self.generate_embeddings(batch)
            self.embeddings.extend(batch_embeddings)

        logger.info(f"Generated embeddings for {len(self.embeddings)} contexts")
    def get_query_embedding(self, query: str):
        query_embedding = self.embed_model.encode(
            sentences=[query],
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return query_embedding[0].tolist()
    def clear(self):
        self.embeddings.clear()
        self.contexts.clear()
        logger.info("Cleared all embeddings and contexts")