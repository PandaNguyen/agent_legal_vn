from crewai import BaseLLM
from typing import Any, Dict, List, Optional, Union
import requests

class HuggingFaceRouterLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        endpoint: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        resolved_model = model[len("huggingface/") :] if model.lower().startswith("huggingface/") else model
        super().__init__(model=resolved_model, temperature=temperature)
        self.api_key = api_key
        self.endpoint = endpoint
        self.max_tokens = max_tokens

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
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        if tools and self.supports_function_calling():
            payload["tools"] = tools

        stream_callback = kwargs.get("stream_callback", None)
        if stream_callback:
            payload["stream"] = True

        response = requests.post(
            self.endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=120,
            stream=bool(stream_callback)
        )
        if not response.ok:
            body = response.text[:2000]
            raise requests.HTTPError(
                f"Hugging Face router request failed: status={response.status_code}, model={self.model}, body={body}",
                response=response,
            )

        if stream_callback:
            import json
            full_content = ""
            for line in response.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data: ') and decoded != 'data: [DONE]':
                        try:
                            data = json.loads(decoded[6:])
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content")
                                if content and isinstance(content, str):
                                    full_content += content
                                    stream_callback(content)
                        except Exception:
                            pass
            return full_content
        else:
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def supports_function_calling(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 32768
