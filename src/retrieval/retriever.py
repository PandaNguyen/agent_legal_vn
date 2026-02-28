from typing import Optional
from src.indexing.qdrant_vdb import QdrantVDB
from src.embeddings.embed_data import Embeddata
from config.settings import settings
from pydantic import BaseModel, Field
class TextNode(BaseModel):
    text: str
    metadata: dict | None = Field(default=None)
class NodeWithScore(BaseModel):
    score: float
    node: TextNode
class Retriever:
    def __init__(self,
                 vector_db: QdrantVDB,
                 embed_data: Embeddata,
                 top_k: int = None):
        self.vector_db = vector_db
        self.embed_data = embed_data
        self.top_k = top_k or settings.top_k
        # Auto-connect to Qdrant if not already done
        if self.vector_db.client is None:
            self.vector_db.initialize_client()

    def search(self,
               query: str,
               top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        # QdrantVDB.search() nhận query string và embed_data để tự generate embeddings
        search_result = self.vector_db.search(query, embed_data=self.embed_data, top_k=top_k)

        node_with_scores = []
        for result in search_result:
            metadata = {
                k: v for k, v in result.items()
                if k not in {"text", "score"}
            }
            node_with_scores.append(NodeWithScore(
                score=result["score"],
                node=TextNode(
                    text=result["text"],
                    metadata=metadata
                )
            ))
        # print(node_with_scores)

        return node_with_scores
        
    def get_context(self,
                    query:str,
                    top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        search_result = self.search(query, top_k=top_k)
        # search() trả về List[dict] với keys: score, text, doc_number, unit_title, ...
        return [node_with_score.node.text for node_with_score in search_result]

    def get_combined_context(self,
                    query:str,
                    top_k: Optional[int] = None):
        context = self.get_context(query, top_k=top_k)
        return "\n\n---\n\n".join(context)
    
    def search_with_score(self,
                        query:str,
                        top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.top_k
        node_with_scores = self.search(query, top_k=top_k)
        results = []
        for node_with_score in node_with_scores:
            results.append({
                "context": node_with_score.node.text,
                "score": node_with_score.score,
                "metadata": node_with_score.node.metadata
            })
        return results
    
    def get_citation(self,
                    query:str,
                    top_k: Optional[int] = None,
                    snippet_chars: int = 300):
        if top_k is None:
            top_k = self.top_k
        results = self.search_with_score(query, top_k)
        citations = []
        for rank, item in enumerate(results, start=1):
            context = (item.get("context") or "").strip()
            snippet = ""
            if context:
                snippet = context[:snippet_chars]
                if len(context) > snippet_chars:
                    snippet += "..."
            citations.append({
                "rank": rank,
                "score": item.get("score"),
                "metadata": item.get("metadata", {}),
                "snippet": snippet
            })    
        return citations
