from typing import Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from firecrawl import FirecrawlApp
from config.settings import settings
import os

class FirecrawlSearchInput(BaseModel):
    query: str = Field(..., description="The search query to lookup on the web")
    limit: int = Field(..., description="The maximum number of results to fetch")


class FirecrawlSearchTool(BaseTool):
    name: str = "Firecraw Web Search"
    description: str = (
        "Search the web using Firecrawl and return a concise list of results "
        "(title, URL, and short description snippet)."
    )
    args_schema: Type[BaseModel] = FirecrawlSearchInput

    def _run(self, query: str, limit: int = 3) -> str:
        api_key = settings.firecrawl_api_key
        if not api_key:
            raise ValueError("Firecrawl API key is required")
        try:
            app = FirecrawlApp(api_key=api_key)
            response = app.search(query, limit=limit)

            # Firecrawl v2 returns SearchData with `.web` (list[SearchResultWeb])
            results_list = getattr(response, "web", None) or getattr(response, "data", None)

            if not isinstance(results_list, list) or not results_list:
                return "No relevant web search results found"

            search_content = []
            for result in results_list:
                # Support both object-style (v2) and dict-style (v1) results
                if isinstance(result, dict):
                    url = result.get("url", "No url")
                    title = result.get("title", "No title")
                    description = (result.get("description") or "").strip()
                else:
                    url = getattr(result, "url", "No url")
                    title = getattr(result, "title", "No title")
                    description = (getattr(result, "description", "") or "").strip()

                snippet = description[:1000] if description else "[No description available]"
                search_content.append(f"Title: {title}\nURL: {url}\nContent: {snippet}")

            return "\n\n---\n\n".join(search_content) if search_content else "No relevant web search results found"
        except Exception as e:
            return f"Web search unavailable: {str(e)}"