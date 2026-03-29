from typing import Any, Dict, List, Optional, Union

import requests
from crewai import BaseLLM

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM(BaseLLM):
    """LLM wrapper for OpenRouter – compatible with crewAI BaseLLM."""

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ):
        # Strip the "openrouter/" prefix if present so the raw model id is sent
        resolved_model = (
            model[len("openrouter/"):] if model.lower().startswith("openrouter/") else model
        )
        super().__init__(model=resolved_model, temperature=temperature)
        self.api_key = api_key
        self.max_tokens = max_tokens
        # Optional HTTP-Referer / X-Title headers (recommended by OpenRouter)
        self.site_url = site_url
        self.site_name = site_name

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def call(
        self,
        messages: Union[str, List[Dict[str, str]]],
        tools: Optional[List[dict]] = None,
        callbacks: Optional[List[Any]] = None,
        available_functions: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Union[str, Any]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]

        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools and self.supports_function_calling():
            payload["tools"] = tools

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
        )

        if not response.ok:
            body = response.text[:2000]
            raise requests.HTTPError(
                f"OpenRouter request failed: status={response.status_code}, "
                f"model={self.model}, body={body}",
                response=response,
            )

        result = response.json()
        return result["choices"][0]["message"]["content"]

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 131072
