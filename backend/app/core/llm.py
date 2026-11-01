"""LLM 客户端（异步、单例）"""
from __future__ import annotations

from functools import lru_cache

from .config import config


class LLMNotConfigured(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client():
    """返回 AsyncOpenAI 客户端；未配置 Key 时抛 LLMNotConfigured。"""
    if not config.llm.configured:
        raise LLMNotConfigured(
            "未配置 LLM API Key，请在 backend/.env 中设置 OPENAI_API_KEY"
        )
    from openai import AsyncOpenAI

    return AsyncOpenAI(
        api_key=config.llm.api_key,
        base_url=config.llm.api_base,
        timeout=config.llm.timeout,
    )
