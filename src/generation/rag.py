from typing import Optional
from loguru import logger
from crewai import LLM
from pydantic import BaseModel
from src.retrieval.retriever import Retriever
from config.settings import settings
class ChatMessage(BaseModel):
    role: str
    content: str
class RAG:
    def __init__(
        self,
        retriever: Retriever,
        llm_model: str = None,
        gemini_api_key: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        self.retriever = retriever
        self.llm_model = llm_model or settings.llm_model
        self.gemini_api_key = gemini_api_key or settings.gemini_api_key
        self.temperature = temperature or settings.temperature
        self.max_tokens = max_tokens or settings.max_tokens
        self.llm = self._setup_llm()
        self.system_message = ChatMessage(
            role="system",
            content="Bạn là một trợ lý hữu ích trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
                   "Luôn dựa câu trả lời của bạn trên thông tin đã cho và nêu rõ khi bạn không biết điều gì đó."
        )
        self.prompt_template = (
            "CONTEXT:\n"
            "{context}\n"
            "---------------------\n"
            "Dựa trên thông tin ngữ cảnh ở trên, vui lòng trả lời câu hỏi sau. "
            "Nếu ngữ cảnh không chứa đủ thông tin để trả lời câu hỏi, hoặc "
            "ngay cả khi bạn biết câu trả lời nhưng nó không liên quan đến ngữ cảnh được cung cấp, "
            "hãy nêu rõ rằng bạn không biết và giải thích thông tin nào còn thiếu.\n\n"
            "QUESTION: {query}\n"
            "ANSWER: "
        )

    def _setup_llm(self):
        if not self.gemini_api_key:
            raise ValueError(
                "Gemini API key is required. Set it in settings.py or pass it as a parameter."
            )
        llm = LLM(
            model=self.llm_model,
            api_key=self.gemini_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        logger.info(f"Initialized LLM: {self.llm_model}")
        return llm
    def generate_context(self, query: str, top_k: Optional[int] = None):
        context = self.retriever.get_combined_context(query, top_k=top_k)
        return context

    def query(self, query: str, top_k: Optional[int] = None):
        context = self.generate_context(query, top_k=top_k)
        prompt = self.prompt_template.format(context=context, query=query)
        return self.llm.call(f"{self.system_message.content}\n\n{prompt}")

    def get_detailed_response(self, query: str, top_k: Optional[int] = None):
        retrieval_result = self.retriever.search_with_score(query=query, top_k=top_k)

        response = self.query(query=query, top_k=top_k)

        return {
            "response": response,
            "source": retrieval_result,
            "query": query,
            "top_k": top_k,
            "model": self.llm_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template
        logger.info(f"Set prompt template: {prompt_template}")

    def set_system_message(self, content: str):
        self.system_message = ChatMessage(role="system", content=content)
        logger.info(f"Set system message: {content}")
    