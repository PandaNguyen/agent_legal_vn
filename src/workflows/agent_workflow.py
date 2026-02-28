from typing import Optional, Any, List
from loguru import logger
from crewai import LLM
from crewai.flow.flow import Flow, start, listen, router, or_
from pydantic import BaseModel

from src.tools.firecrawl_search_tool import FirecrawlSearchTool
from src.retrieval.retriever import Retriever, NodeWithScore
from src.generation.rag import RAG
from config.settings import settings
from .event import RetrieveEvent, EvaluateEvent, WebSearchEvent, SynthesizeEvent

# ─── Prompt templates ─────────────────────────────────────────────────────────
ROUTER_EVALUATION_TEMPLATE = (
    """Bạn là một người đánh giá chất lượng cho các phản hồi RAG. Nhiệm vụ của bạn là xác định xem phản hồi được cung cấp có trả lời đầy đủ câu hỏi của người dùng hay không.

USER QUESTION:
{query}

RAG RESPONSE:
{rag_response}

EVALUATION CRITERIA:
- Phản hồi có trực tiếp trả lời câu hỏi của người dùng không?
- Phản hồi có nhất quán về mặt thông tin và được trình bày rõ ràng, có cấu trúc không?
- Phản hồi có đủ chi tiết để hữu ích không?
- Nếu phản hồi nói "Tôi không biết" hoặc tương tự, có phải vì ngữ cảnh thực sự thiếu thông tin không?

Vui lòng đánh giá chất lượng phản hồi và trả lời bằng một trong hai lựa chọn:
- "GOOD" - nếu phản hồi trả lời đầy đủ câu hỏi
- "BAD" - nếu phản hồi chưa đầy đủ, không rõ ràng hoặc không trả lời đúng câu hỏi

QUAN TRỌNG: Chỉ trả lời DUY NHẤT MỘT TỪ viết IN HOA: GOOD hoặc BAD. Không thêm dấu câu hoặc nội dung nào khác.

Your evaluation (GOOD or BAD):"""
)

QUERY_OPTIMIZATION_TEMPLATE = (
    """Tối ưu hóa truy vấn sau cho tìm kiếm trên web để nhận được kết quả phù hợp và chính xác nhất.

Original Query: {query}

Hướng dẫn:
- Làm cho truy vấn cụ thể và dễ tìm kiếm hơn
- Thêm các từ khóa liên quan giúp tìm được nguồn đáng tin cậy
- Giữ ngắn gọn nhưng đầy đủ
- Tập trung vào nhu cầu thông tin cốt lõi

Optimized Query:"""
)

SYNTHESIS_TEMPLATE = (
    """Bạn là một hệ thống tổng hợp câu trả lời. Hãy tạo một câu trả lời toàn diện và chính xác dựa trên thông tin có sẵn.

USER QUESTION:
{query}

RAG RESPONSE (from document knowledge):
{rag_response}

WEB SEARCH RESULTS (additional context):
{web_results}

INSTRUCTIONS:
- Tổng hợp thông tin từ cả hai nguồn để cung cấp câu trả lời đầy đủ nhất
- Ưu tiên thông tin từ các nguồn đáng tin cậy
- Nếu có mâu thuẫn, hãy nêu rõ
- Chỉ rõ thông tin nào đến từ tìm kiếm web và thông tin nào từ tài liệu
- Nếu kết quả web trống, hãy tinh chỉnh và cải thiện phản hồi RAG

SYNTHESIZED RESPONSE:"""
)


# ─── State ────────────────────────────────────────────────────────────────────
class AgentState(BaseModel):
    """Mutable state shared across all flow steps."""
    query: str = ""
    top_k: Optional[int] = 3
    # Populated by flow steps — used instead of event passing
    retrieved_nodes: List[Any] = []
    rag_response: str = ""
    web_results: str = ""
    use_web_results: bool = False


class AgentWorkflow(Flow[AgentState]):
    def __init__(
        self,
        retriever: Retriever,
        rag_system: RAG,
        firecrawl_apikey: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.retriever = retriever
        self.rag = rag_system

        self.gemini_api_key = gemini_api_key or settings.gemini_api_key
        self.firecrawl_apikey = firecrawl_apikey or settings.firecrawl_api_key
        self.qdrant_url = qdrant_url or settings.qdrant_url
        self.qdrant_api_key = qdrant_api_key or settings.qdrant_api_key

        self.llm = LLM(
            model=settings.llm_model,
            api_key=self.gemini_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    @start()
    def retrieve(self):
        query = self.state.query
        top_k = self.state.top_k

        if not query:
            raise ValueError("Query is required")

        logger.info(f"[retrieve] query='{query}', top_k={top_k}")
        retrieved_nodes = self.retriever.search(query, top_k=top_k)
        logger.info(f"[retrieve] got {len(retrieved_nodes)} nodes")

        # Store in state so downstream steps can access
        self.state.retrieved_nodes = retrieved_nodes

    # ── Step 2: RAG generation ────────────────────────────────────────────────
    @listen(retrieve)
    def generate_rag_response(self):
        query = self.state.query
        logger.info(f"[generate_rag_response] generating answer for '{query}'")

        rag_response = self.rag.query(query, top_k=self.state.top_k)
        self.state.rag_response = rag_response or ""
        logger.info("[generate_rag_response] done")

    # ── Step 3: Evaluate quality (router) ─────────────────────────────────────
    @router(generate_rag_response)
    def evaluate(self):
        query = self.state.query
        rag_response = self.state.rag_response

        logger.info(f"[evaluate] checking response quality")
        prompt = ROUTER_EVALUATION_TEMPLATE.format(query=query, rag_response=rag_response)
        response_text = self.llm.call(prompt)

        evaluation = (response_text or "").strip().upper()
        route = "synthesize" if evaluation.startswith("GOOD") else "web_search"
        logger.info(f"[evaluate] result='{evaluation[:20]}' → route='{route}'")
        return route

    # ── Step 4a: Web search (if BAD) ──────────────────────────────────────────
    @listen("web_search")
    def perform_web_search(self):
        query = self.state.query
        logger.info(f"[perform_web_search] query='{query}'")

        try:
            opt_prompt = QUERY_OPTIMIZATION_TEMPLATE.format(query=query)
            optimized = (self.llm.call(opt_prompt) or query).strip()
            search_results = FirecrawlSearchTool()._run(query=optimized, limit=3)
            logger.info("[perform_web_search] done")
        except Exception as e:
            logger.error(f"[perform_web_search] failed: {e}")
            search_results = "Web search unavailable due to technical issues."

        self.state.web_results = search_results
        self.state.use_web_results = True

    # ── Step 5: Synthesize final answer ───────────────────────────────────────
    @listen(or_("synthesize", perform_web_search))
    def synthesize_response(self):
        query = self.state.query
        rag_response = self.state.rag_response
        web_results = self.state.web_results
        use_web = self.state.use_web_results

        logger.info(f"[synthesize_response] use_web={use_web}")

        if use_web and web_results:
            prompt = SYNTHESIS_TEMPLATE.format(
                query=query, rag_response=rag_response, web_results=web_results
            )
            answer = self.llm.call(prompt)
            result = {
                "answer": answer,
                "rag_response": rag_response,
                "web_search_used": True,
                "web_results": web_results,
                "query": query,
            }
        else:
            refine_prompt = (
                f"Improve and refine the following response to make it more helpful and comprehensive:\n\n"
                f"Original Response: {rag_response}\n\nRefined Response:"
            )
            answer = self.llm.call(refine_prompt)
            result = {
                "answer": answer,
                "rag_response": rag_response,
                "web_search_used": False,
                "web_results": None,
                "query": query,
            }

        logger.info("[synthesize_response] done")
        return result

    # ── Public API ────────────────────────────────────────────────────────────
    def run_workflow(self, query: str, top_k: Optional[int] = None) -> dict:
        """Run the full workflow and return the final answer dict."""
        # Reset volatile state before each run to prevent leakage between queries
        self.state.retrieved_nodes = []
        self.state.rag_response = ""
        self.state.web_results = ""
        self.state.use_web_results = False

        try:
            result = self.kickoff(inputs={"query": query, "top_k": top_k or settings.top_k})
            # kickoff() returns the output of the last executed step
            if isinstance(result, dict):
                return result
            return {"answer": str(result), "query": query}
        except Exception as e:
            logger.error(f"[run_workflow] error: {e}")
            return {
                "answer": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "rag_response": None,
                "web_search_used": False,
                "web_results": None,
                "query": query,
                "error": str(e),
            }