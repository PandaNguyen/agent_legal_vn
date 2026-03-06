from __future__ import annotations

from typing import Optional

from crewai import LLM

from paralegal_agent.config.config import settings
from paralegal_agent.provider.Huggingface.qwen import HuggingFaceRouterLLM


def _is_gemini_model(model_name: str) -> bool:
    name = model_name.lower()
    return name.startswith("gemini") or name.startswith("google/")


def _is_ollama_model(model_name: str) -> bool:
    return model_name.lower().startswith("ollama/")


def _is_huggingface_model(model_name: str) -> bool:
    return model_name.lower().startswith("huggingface/")


def create_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
):
    model_name = model_name or settings.llm_model
    temperature = settings.temperature if temperature is None else temperature
    max_tokens = settings.max_tokens if max_tokens is None else max_tokens

    kwargs = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if _is_ollama_model(model_name):
        kwargs["base_url"] = settings.ollama_base_url
    elif _is_huggingface_model(model_name):
        if not settings.huggingface_api_key:
            raise ValueError("Hugging Face API key is required. Set HUGGINGFACE_API_KEY or HF_TOKEN.")
        return HuggingFaceRouterLLM(
            model=model_name,
            api_key=settings.huggingface_api_key,
            endpoint=settings.huggingface_router_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif _is_gemini_model(model_name):
        kwargs["api_key"] = settings.gemini_api_key

    return LLM(**kwargs)
