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

        stream_callback = kwargs.get("stream_callback", None)
        if stream_callback:
            payload["stream"] = True

        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=120,
            stream=bool(stream_callback),
        )

        if not response.ok:
            body = response.text[:2000]
            raise requests.HTTPError(
                f"OpenRouter request failed: status={response.status_code}, "
                f"model={self.model}, body={body}",
                response=response,
            )

        if stream_callback:
            import json
            full_content = ""
            current_reasoning = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                        try:
                            # It could be missing data?
                            data = json.loads(decoded[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                
                                # Process reasoning tokens if present
                                reasoning = delta.get("reasoning")
                                if reasoning and isinstance(reasoning, str):
                                    current_reasoning += reasoning
                                    # Stream reasoning enclosed in italics so user sees thinking process
                                    # But since markdown redraws the whole string, we could just append it dynamically.
                                    # Actually, Streamlit will just render the latest string we send it.
                                    # If we want to show reasoning, we should send full_content + " *Suy nghĩ...*" or something.
                                    # But better to just display the reasoning!
                                    stream_callback(reasoning)
                                
                                content = delta.get("content")
                                if content and isinstance(content, str):
                                    full_content += content
                                    stream_callback(content)
                        except Exception as e:
                            print(f"[DEBUG SSE Error] {e} on line: {decoded}")
            return full_content
        else:
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 131072
