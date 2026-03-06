from pydantic import BaseModel, Field
from typing import Any, List, Optional
class TextNode(BaseModel):
    text: str
    metadata: dict | None = Field(default=None)
class NodeWithScore(BaseModel):
    score: float
    node: TextNode

class RetrieveEvent(BaseModel):
    """Event containing retrieved nodes from vector database."""
    retrieved_nodes: List[NodeWithScore]
    query: str

class RAGResponseEvent(BaseModel):
    """Event containing the generated RAG response."""
    rag_response: str
    query: str
    context: str
class WebSearchEvent(BaseModel):
    """Event for web search when RAG response is insufficient."""
    rag_response: str
    query: str
    search_results: Optional[str] = None
class SynthesizeEvent(BaseModel):
    """Event for final response synthesis."""
    rag_response: str
    retrieved_nodes: List[NodeWithScore]
    query: str
    web_search_results: Optional[str] = None
    use_web_results: bool = False