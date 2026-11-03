"""接口的请求 / 响应模型。

单独放一个模块，是为了让路由文件和「数据形状」解耦：
前端契约变更时只改这里，路由逻辑不用动。
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    question: str = ""
    answer: str = ""


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: List[ChatTurn] = Field(default_factory=list)
    use_rerank: bool = True
    doc_id: Optional[str] = None
    # auto：按问题类型自动选择；fast：检索片段后回答；precise：选页 + 整页精读
    mode: Literal["auto", "fast", "precise"] = "auto"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    doc_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=30)


class ConversationCreate(BaseModel):
    title: str = ""


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=60)


class MessageIds(BaseModel):
    ids: List[int] = Field(min_length=1, max_length=100)


class MessageAppend(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: List[dict] = Field(default_factory=list)
    confidence: Optional[float] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    confidence: float
    retrieval_count: int
    elapsed_ms: int
    cached: bool = False
    checks: List[dict] = Field(default_factory=list)


__all__ = [
    "ChatTurn",
    "ConversationCreate",
    "ConversationRename",
    "MessageAppend",
    "MessageIds",
    "QueryRequest",
    "QueryResponse",
    "SearchRequest",
]
