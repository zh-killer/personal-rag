"""答案生成：单一路径的 prompt 组装 + 流式输出 + 来源清单

修复点：
- 旧版 assembler 先拼了一个完整 system prompt，generate 又把它塞进 user 消息，
  规则重复两遍、知识片段被降级成用户输入；现在 system 只放规则，资料只在 user 里出现一次；
- 上下文按字符预算截断（中文约 1.7 字符/token），并给每个片段标 [n] 与页码；
- 流式与非流式共用同一套 prompt。
"""
from __future__ import annotations

import logging
from typing import AsyncIterator, Dict, List, Optional

from ..core.config import config
from ..core.llm import get_client
from .highlight import matched_terms

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个严谨的个人知识库助手。

规则：
1. 只依据「参考资料」回答，不要编造资料里没有的内容；
2. 资料不足时直接说明「根据现有资料无法确定」，并指出缺少什么信息；
3. 引用资料时在句末标注编号，例如 [1]、[2][3]；
4. 回答用 Markdown，简洁、分点，先说结论；
5. 如果用户问的是对话本身（例如「我刚才问了什么」「上一个问题是什么」），
   可以依据「对话历史」直接回答，不需要查资料；
6. 与知识库无关的问题，礼貌说明只能回答知识库内容。"""

MAX_CONTEXT_CHARS = 6000
MAX_CHUNK_CHARS = 900

PAGE_SYSTEM_PROMPT = """你是严谨的文档精读助手。用户提供了若干页的完整原文，请直接依据原文回答。

规则：
1. 只使用给出的页面原文，不要引入外部知识；
2. 回答要具体：涉及数字、单词、定义、列表时照抄原文，不要概括或省略；
3. 句末标注页面编号，例如 [1]；引用关键原句时用引号标出；
4. 如果原文里确实没有，明确说明「这几页里没有」，不要猜；
5. 用 Markdown，先给结论，再给依据。"""


class AnswerGenerator:
    def __init__(self) -> None:
        self.max_context_chars = MAX_CONTEXT_CHARS

    # ── prompt ───────────────────────────────────────────
    def build_messages(
        self, query: str, chunks: List[Dict], history: Optional[List[dict]] = None
    ) -> List[Dict]:
        context_parts: List[str] = []
        used_chars = 0
        for index, chunk in enumerate(chunks, start=1):
            source = (chunk.get("metadata") or {}).get("source", "未知文档")
            page = (chunk.get("metadata") or {}).get("page")
            header = f"[{index}] 来源：{source}" + (f"（第 {page} 页）" if page else "")
            body = str(chunk.get("content", ""))[:MAX_CHUNK_CHARS]
            block = f"{header}\n{body}"
            if used_chars + len(block) > self.max_context_chars:
                break
            context_parts.append(block)
            used_chars += len(block)

        history_text = self._format_history(history)
        user_parts = ["## 参考资料", "\n\n".join(context_parts) or "（无）"]
        if history_text:
            user_parts += ["## 对话历史", history_text]
        user_parts += ["## 用户问题", query]

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]

    @staticmethod
    def _format_history(history: Optional[List[dict]]) -> str:
        if not history:
            return ""
        lines: List[str] = []
        for turn in history[-config.history_turns:]:
            question = str(turn.get("question", "")).strip()
            answer = str(turn.get("answer", "")).strip()
            if question:
                lines.append(f"用户：{question}")
            if answer:
                lines.append(f"助手：{answer[:400]}")
        return "\n".join(lines)

    # ── 生成 ─────────────────────────────────────────────
    async def answer(
        self, query: str, chunks: List[Dict], history: Optional[List[dict]] = None
    ) -> str:
        client = get_client()
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=self.build_messages(query, chunks, history),
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream(
        self, query: str, chunks: List[Dict], history: Optional[List[dict]] = None
    ) -> AsyncIterator[str]:
        client = get_client()
        stream = await client.chat.completions.create(
            model=config.llm.model,
            messages=self.build_messages(query, chunks, history),
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            stream=True,
        )
        async for event in stream:
            if not event.choices:
                continue
            delta = event.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    # ── 精准模式：整页原文 ────────────────────────────────
    def build_page_messages(
        self, query: str, pages: List[Dict], history: Optional[List[dict]] = None
    ) -> List[Dict]:
        blocks = []
        for index, page in enumerate(pages, start=1):
            text = str(page.get("text", ""))[:2600]
            blocks.append(
                f"[{index}] 文档：{page.get('source', '')}  第 {page.get('page')} 页\n{text}"
            )
        context = "\n\n---\n\n".join(blocks)[:9000]
        history_text = self._format_history(history)
        parts = ["## 页面原文", context or "（无）"]
        if history_text:
            parts += ["## 对话历史", history_text]
        parts += ["## 用户问题", query]
        return [
            {"role": "system", "content": PAGE_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(parts)},
        ]

    async def answer_pages(
        self, query: str, pages: List[Dict], history: Optional[List[dict]] = None
    ) -> str:
        client = get_client()
        response = await client.chat.completions.create(
            model=config.llm.model,
            messages=self.build_page_messages(query, pages, history),
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
        )
        return response.choices[0].message.content or ""

    async def stream_pages(
        self, query: str, pages: List[Dict], history: Optional[List[dict]] = None
    ) -> AsyncIterator[str]:
        client = get_client()
        stream = await client.chat.completions.create(
            model=config.llm.model,
            messages=self.build_page_messages(query, pages, history),
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            stream=True,
        )
        async for event in stream:
            if not event.choices:
                continue
            content = getattr(event.choices[0].delta, "content", None)
            if content:
                yield content

    # ── 来源 ─────────────────────────────────────────────
    @staticmethod
    def sources(chunks: List[Dict], query: str = "") -> List[Dict]:
        sources: List[Dict] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk.get("metadata") or {}
            content = str(chunk.get("content", ""))
            sources.append(
                {
                    "index": index,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "source": metadata.get("source", "未知文档"),
                    "page": metadata.get("page"),
                    "snippet": content[:160].replace("\n", " "),
                    "terms": matched_terms(query, content),
                }
            )
        return sources


# 全局单例
generator = AnswerGenerator()
