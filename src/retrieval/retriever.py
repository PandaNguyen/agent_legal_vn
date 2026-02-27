from typing import Optional
from src.indexing.qdrant_vdb import QdrantVDB
from src.embeddings.embed_data import Embeddata
from config.settings import settings

class Retriever:
    def __init__(self,
                 vector_db: QdrantVDB,
                 embed_data: Embeddata,
                 top_k: int = None):
        self.vector_db = vector_db
        self.embed_data = embed_data
        self.top_k = top_k or settings.top_k
    
    def search(self,
               query: str,
               top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        # QdrantVDB.search() nhận query string và embed_data để tự generate embeddings
        search_result = self.vector_db.search(query, embed_data=self.embed_data, top_k=top_k)
        return search_result
    def get_combined_context(self,
                    query:str,
                    top_k: Optional[int] = None):
        context = self.get_context(query, top_k=top_k)
        return "\n\n---\n\n".join(context)
    def get_context(self,
                    query:str,
                    top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        search_result = self.search(query, top_k=top_k)
        # search() trả về List[dict] với keys: score, text, doc_number, unit_title, ...
        return [hit["text"] for hit in search_result]
    def get_citation(self,
                    query:str,
                    top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        search_result = self.search(query, top_k=top_k)
        # search() trả về List[dict] với keys: score, text, doc_number, unit_title, ...
        return search_result
