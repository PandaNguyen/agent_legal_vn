from crewai import LLM
from paralegal_agent.config.config import settings
class Phi3_5LLM:
    def __init__(self):
        self.llm = LLM(
            model="phi3.5",
            base_url="http://localhost:11434",
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
    def get_llm(self):
        return self.llm