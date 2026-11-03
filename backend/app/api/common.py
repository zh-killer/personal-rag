"""路由之间共用的小工具。

注意：这里只放无 I/O 或只读索引的纯函数，方便单测直接 import。
"""
from __future__ import annotations

import json
from typing import List, Optional

from ..core.config import config
from ..deps import mixed_index, query_router, reranker, retriever, rewriter
from ..indexing.tokenizer import extract_terms
from ..schemas import ChatTurn


def sse(event: dict) -> str:
    """SSE 帧：内容整体做 JSON 转义，换行不会破坏协议。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def history_payload(turns: List[ChatTurn]) -> List[dict]:
    """只保留最近 N 轮，避免长对话把上下文窗口打满。"""
    return [turn.model_dump() for turn in turns[-config.history_turns:]]


def search_snippet(content: str, query: str, width: int = 160) -> str:
    """截取命中词附近的文本，方便在结果里直接看到上下文。"""
    text = " ".join(content.split())
    lowered = text.lower()
    position = -1
    for term in extract_terms(query):
        found = lowered.find(term)
        if found >= 0:
            position = found
            break
    if position < 0:
        return text[: width * 2]
    start = max(0, position - width // 2)
    end = min(len(text), position + width)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def meta_answer() -> str:
    """知识库元问题（「库里有哪些文档」）直接查台账，不走向量检索。"""
    docs = mixed_index.list_docs()
    if not docs:
        return (
            "知识库还是空的。\n\n"
            f"支持上传：{'、'.join(config.supported_formats)}\n\n"
            "到「上传」标签页拖入 PDF / Markdown / TXT 即可开始。"
        )
    lines = [f"知识库当前有 **{len(docs)}** 个文档：\n"]
    for doc in docs[:50]:
        pages = f"，{doc['page_count']} 页" if doc.get("page_count") else ""
        lines.append(f"- **{doc['source']}** — {doc['chunk_count']} 个片段{pages}")
    lines.append("\n直接提问即可，我会基于这些文档回答并标注 [编号] 来源。")
    return "\n".join(lines)


async def prepare_chunks(
    question: str,
    history: List[dict],
    use_rerank: bool,
    top_k: int,
    doc_id: Optional[str] = None,
) -> tuple[List[dict], float]:
    """路由 → 改写 → 多路召回 → 重排，返回 (片段, 置信度)。"""
    decision = query_router.route(question, has_history=bool(history))
    variants = await rewriter.rewrite(question, history, use_llm=decision.use_rewrite)
    result = await retriever.retrieve(
        question, variants, top_k=top_k or decision.top_k, doc_id=doc_id
    )
    chunks = result.chunks
    if use_rerank and chunks:
        chunks = await reranker.rerank(question, chunks)
    return chunks, result.confidence


__all__ = [
    "history_payload",
    "meta_answer",
    "prepare_chunks",
    "search_snippet",
    "sse",
]
