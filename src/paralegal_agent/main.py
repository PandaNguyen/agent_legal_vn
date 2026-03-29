from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
from crewai.flow import Flow, listen, start, router, or_
from loguru import logger
from paralegal_agent.crews.rag import QdrantRAGCrew
from paralegal_agent.crews.evaluate import EvaluateCrew
from paralegal_agent.event import RetrieveEvent, RAGResponseEvent, NodeWithScore, WebSearchEvent, SynthesizeEvent
from paralegal_agent.retrieval.retriever import Retriever
from paralegal_agent.config.config import settings
from paralegal_agent.indexing.qdrant_vdb import QdrantVDB
from paralegal_agent.embeddings.embed_data import Embeddata
from paralegal_agent.tools.firecrawl_search_tool import FirecrawlSearchTool
from paralegal_agent.provider.llm_factory import create_llm
from paralegal_agent.cache.query_cache import QueryCacheLookup
class AgentState(BaseModel):
    query: str = ""
    top_k: Optional[int] = 3
    retrieved_nodes: List[NodeWithScore] = []
    rag_response: str = ""
    cache_threshold: float = 0.85
    corpus_path: str = settings.docs_path
    stream_callback: Optional[Any] = None


def extract_citations(nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
    """
    Trích xuất citation từ NodeWithScore.
    Metadata lồng nhau dạng: node.metadata = {"metadata": {...}}
    """
    citations = []
    for i, node in enumerate(nodes):
        raw_meta = node.node.metadata or {}
        meta = raw_meta.get("metadata", raw_meta)  # unwrap lớp lồng nhau

        citations.append({
            "index":             i + 1,
            "score":             round(node.score, 3),
            "text_snippet":      node.node.text[:300] + "..." if len(node.node.text) > 300 else node.node.text,
            "doc_number":        meta.get("doc_number", ""),
            "doc_title":         meta.get("doc_title", ""),
            "doc_type":          meta.get("doc_type", ""),
            "legal_field":       meta.get("legal_field", ""),
            "issuing_authority": meta.get("issuing_authority", ""),
            "signer":            meta.get("signer", ""),
            "issue_date":        meta.get("issue_date", ""),
            "source_url":        meta.get("source_url", ""),
            "unit_title":        meta.get("unit_title", ""),
        })
    return citations


class AgentFlow(Flow[AgentState]):
    def __init__(
        self,
        retriever: Retriever,
        gemini_api_key: Optional[str] = None,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        cache_threshold: float = 0.85,
        corpus_path: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retriever        = retriever
        self.gemini_api_key   = gemini_api_key or settings.gemini_api_key
        self.llm_model        = llm_model   if llm_model   is not None else settings.llm_model
        self.temperature      = temperature if temperature is not None else settings.temperature
        self.max_tokens       = max_tokens  if max_tokens  is not None else settings.max_tokens
        self.cache_threshold  = cache_threshold
        self.corpus_path      = corpus_path or settings.docs_path
        self._cache_lookup    = QueryCacheLookup(
            client     = retriever.vector_db.client,
            embed_data = retriever.embed_data,
            threshold  = self.cache_threshold,
            # corpus_path: None → QueryCacheLookup uses __file__-relative default
            corpus_path = corpus_path if corpus_path else None,
        )
        logger.info(f"Using LLM model: {self.llm_model}, temperature: {self.temperature}, max_tokens: {self.max_tokens}")

    def _make_llm(self):
        return create_llm(
            model_name=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def _call_llm(self, prompt: str, stream_callback=None) -> str:
        llm = self._make_llm()
        try:
            import inspect
            sig = inspect.signature(llm.call)
            if "stream_callback" in sig.parameters or "kwargs" in sig.parameters:
                response = llm.call(messages=[{"role": "user", "content": prompt}], stream_callback=stream_callback)
            else:
                response = llm.call(messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            logger.warning(f"Error inspecting llm.call signature, falling back to basic call: {e}")
            response = llm.call(messages=[{"role": "user", "content": prompt}])
        return response.strip() if isinstance(response, str) else str(response).strip()

    @start()
    def retrieve(self) -> RetrieveEvent:
        """Step 0: cache check first, then normal retrieval on miss."""
        query = self.state.query
        top_k = self.state.top_k
        if not query:
            raise ValueError("Query is required")

        # --- Cache lookup ---
        logger.info(f"[Cache] Checking for query: {query}")
        hit = self._cache_lookup.search(query)
        if hit:
            self._cache_hit_context = hit["context"]
            self._cache_hit_score   = hit["score"]
            self._cache_hit_citations = hit.get("citations", [])
            logger.info(f"[Cache] HIT  score={hit['score']:.4f} — skipping retrieval")
            return RetrieveEvent(retrieved_nodes=[], query=query)

        # --- Cache MISS: normal vector retrieval ---
        self._cache_hit_context = None
        self._cache_hit_score   = 0.0
        self._cache_hit_citations = []
        logger.info("[Cache] MISS — running normal retrieval")
        result = self.retriever.search(query, top_k=top_k)
        self.state.retrieved_nodes = result
        return RetrieveEvent(retrieved_nodes=result, query=query)

    @listen(retrieve)
    def generate_rag_response(self, event: RetrieveEvent) -> RAGResponseEvent:
        query = self.state.query

        # --- Cache HIT: build context from JSON corpus ---
        if self._cache_hit_context:
            logger.info("[Cache] Generating answer from cached law-unit context")
            context = self._cache_hit_context
            self.state.rag_response = "__CACHE_HIT__"  # sentinel, replaced in synthesize
            return RAGResponseEvent(rag_response="__CACHE_HIT__", query=query, context=context)

        # --- Normal RAG ---
        logger.info(f"Generating RAG response for query: {query}")
        context = "\n\n---\n\n".join([node.node.text for node in self.state.retrieved_nodes])
        result  = QdrantRAGCrew(
            llm_model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).crew().kickoff(inputs={"query": query, "context": context})
        self.state.rag_response = result.raw or ""
        return RAGResponseEvent(rag_response=self.state.rag_response, query=query, context=context)

    @router(generate_rag_response)
    def evaluate_response(self, event: RAGResponseEvent) -> str:
        # Cache HIT: skip evaluation, go straight to synthesize
        if self._cache_hit_context:
            logger.info("[Cache] Skipping evaluation — routing to synthesize")
            return "synthesize"
        logger.info(f"Evaluating RAG response for query: {event.query}")
        result     = EvaluateCrew(
            llm_model=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ).crew().kickoff(inputs={"query": event.query, "rag_response": event.rag_response})
        result_str = (result.raw or "").strip().upper().split()[0]
        logger.info(f"Evaluation result: {result_str}")
        return "synthesize" if result_str == "GOOD" else "web_search"

    @listen("web_search")
    async def perform_web_search(self, event: RAGResponseEvent | WebSearchEvent) -> SynthesizeEvent:
        query = self.state.query
        logger.info(f"Performing web search for query: {query}")

        rewrite_prompt = f"""Tối ưu hóa truy vấn sau thành một câu tìm kiếm web ngắn gọn, rõ ràng, hiệu quả.

Câu hỏi gốc: {query}

Yêu cầu:
- Ngắn gọn, cụ thể, dễ tìm kiếm
- Giữ nguyên ngôn ngữ (tiếng Việt)
- Chỉ trả về câu truy vấn, không giải thích thêm

Truy vấn tối ưu:"""

        try:
            new_query = self._call_llm(rewrite_prompt)
            logger.info(f"Rewritten query: {new_query}")
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original: {e}")
            new_query = query

        try:
            search_results = FirecrawlSearchTool().run(query=new_query or query, limit=3)
            logger.info("Web search completed")
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            search_results = "Web search unavailable due to technical issues."

        return SynthesizeEvent(
            rag_response=self.state.rag_response,
            retrieved_nodes=self.state.retrieved_nodes,
            query=query,
            web_search_results=search_results,
            use_web_results=True,
        )

    @listen(or_("evaluate_response", "perform_web_search"))
    async def synthesize_response(self, event: RAGResponseEvent | SynthesizeEvent) -> Dict[str, Any]:
        logger.info("Synthesizing final response")
        query = self.state.query

        # --- Cache HIT: generate directly from corpus law-unit context ---
        if self._cache_hit_context:
            score   = self._cache_hit_score
            context = self._cache_hit_context
            citations = getattr(self, "_cache_hit_citations", [])
            logger.info(f"[Cache] Synthesizing from cached context (score={score:.4f})")
            prompt = f"""Dựa trên các quy định pháp luật dưới đây, hãy trả lời câu hỏi một cách rõ ràng và chính xác bằng tiếng Việt.

Câu hỏi: {query}

Quy định pháp luật liên quan:
{context}

Câu trả lời:"""
            try:
                final_answer = self._call_llm(prompt, stream_callback=getattr(self.state, "stream_callback", None))
            except Exception as e:
                logger.error(f"Cache-hit LLM call failed: {e}")
                final_answer = context
            return {
                "answer":             final_answer,
                "rag_response":       final_answer,
                "web_search_results": None,
                "used_web_results":   False,
                "query":              query,
                "citations":          citations,
                "from_cache":         True,
                "cache_score":        round(score, 4),
                "context":            context
            }

        # --- Normal flow ---
        rag_response       = self.state.rag_response
        web_search_results = getattr(event, "web_search_results", None)
        use_web_results    = getattr(event, "use_web_results", False)

        if use_web_results and web_search_results:
            prompt = f"""Tổng hợp câu trả lời toàn diện dựa trên các nguồn sau.

Câu hỏi: {query}

Phản hồi từ tài liệu (RAG):
{rag_response}

Kết quả tìm kiếm web:
{web_search_results}

Hướng dẫn:
- Kết hợp thông tin từ cả hai nguồn
- Nếu có mâu thuẫn, nêu rõ
- Trả lời bằng tiếng Việt, rõ ràng và có cấu trúc

Câu trả lời tổng hợp:"""
            used_web = True
        else:
            prompt = f"""Cải thiện và hoàn thiện câu trả lời sau để rõ ràng và hữu ích hơn.

Câu hỏi: {query}

Phản hồi RAG gốc:
{rag_response}

Câu trả lời được cải thiện (tiếng Việt):"""
            used_web = False

        try:
            final_answer = self._call_llm(prompt, stream_callback=getattr(self.state, "stream_callback", None))
        except Exception as e:
            logger.error(f"Synthesis failed, falling back to raw RAG: {e}")
            final_answer = rag_response

        citations = extract_citations(self.state.retrieved_nodes)
        logger.info(f"Extracted {len(citations)} citations from retrieved nodes")

        return {
            "answer":              final_answer,
            "rag_response":        rag_response,
            "web_search_results":  web_search_results if used_web else None,
            "used_web_results":    used_web,
            "query":               query,
            "citations":           citations,  
        }


def kickoff():
    vector_db  = QdrantVDB()
    retriever  = Retriever(vector_db=vector_db, embed_data=Embeddata())
    agent_flow = AgentFlow(retriever=retriever)
    agent_flow.kickoff({"query": "Thưa luật sư tôi có đăng ký kết hôn trên pháp luật nhưng nay vợ chồng bỏ nhau theo phong tục tập quán như vậy tôi có được phép kết hôn với người khác không ạ?", "top_k": 1})


def plot():
    vector_db  = QdrantVDB()
    retriever  = Retriever(vector_db=vector_db, embed_data=Embeddata())
    agent_flow = AgentFlow(retriever=retriever)
    agent_flow.plot("./agent_flow_plot.html")


class AgentWorkflow:
    def __init__(
        self,
        retriever: Retriever,
        gemini_api_key: Optional[str]  = None,
        llm_model:      Optional[str]  = None,
        temperature:    Optional[float]= None,
        max_tokens:     Optional[int]  = None,
        cache_threshold: float         = 0.85,
        corpus_path:    Optional[str]  = None,
    ):
        self.flow = AgentFlow(
            retriever       = retriever,
            gemini_api_key  = gemini_api_key or settings.gemini_api_key,
            llm_model       = llm_model   if llm_model   is not None else settings.llm_model,
            temperature     = temperature if temperature is not None else settings.temperature,
            max_tokens      = max_tokens  if max_tokens  is not None else settings.max_tokens,
            cache_threshold = cache_threshold,
            corpus_path     = corpus_path,
        )

    def kickoff(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return self.flow.kickoff(inputs)
        except Exception as e:
            raise Exception(f"An error occurred while kicking off the agent workflow: {e}")


if __name__ == "__main__":
    kickoff()
