"""精准模式：大模型先选页，再通读整页原文回答

fast 模式 = 「向量/BM25 选片段 → 模型只看 5 个片段」，
某个词、某条定义、某页的细节一旦没被召回，模型只能回答「没收录」。

精准模式把「选什么内容」也交给模型：
  1. 混合召回得到候选页（按页聚合，而不是按 500 字片段）；
  2. 模型阅读每页开头摘要，挑出最多 3 页；
  3. 把这 3 页的完整原文交给模型，要求照抄原文并标注页码。
多花 1 次 LLM 调用（选页），换取「细节不再靠运气」。
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..core.config import config
from ..core.llm import get_client
from ..indexing.indexer import mixed_index
from ..indexing.tokenizer import extract_terms
from .highlight import matched_terms
from .retriever import retriever

logger = logging.getLogger(__name__)

MAX_PAGES = 3
CANDIDATE_PAGES = 6


def _strong_terms(question: str) -> List[str]:
    """足够“硬”的检索词：英文/数字整词，或 2 位以上数字；中文二元组太泛，不参与锁定。"""
    terms = []
    for term in extract_terms(question):
        if not term.isascii():
            continue
        if len(term) >= 3 or (term.isdigit() and len(term) >= 2):
            terms.append(term)
    return terms


@dataclass
class PageRef:
    doc_id: str
    source: str
    page: int
    score: float = 0.0
    text: str = ""

    def as_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
        }


class PreciseReader:
    async def select_pages(
        self, question: str, doc_id: Optional[str] = None
    ) -> List[PageRef]:
        result = await retriever.retrieve(question, [question], top_k=24, doc_id=doc_id)

        grouped: Dict[Tuple[str, int], PageRef] = {}
        for chunk in result.chunks:
            metadata = chunk.get("metadata") or {}
            page = int(metadata.get("page") or 0)
            key = (str(metadata.get("doc_id", "")), page)
            if page <= 0 or not key[0]:
                continue
            ref = grouped.get(key)
            if ref is None:
                grouped[key] = PageRef(
                    doc_id=key[0],
                    source=str(metadata.get("source", "")),
                    page=page,
                    score=float(chunk.get("retrieval_score", 0.0)),
                    text=str(chunk.get("content", "")),
                )
            else:
                ref.score = max(ref.score, float(chunk.get("retrieval_score", 0.0)))

        candidates = sorted(
            grouped.values(), key=lambda item: item.score, reverse=True
        )[:CANDIDATE_PAGES]

        # 用整页原文替换片段（同页 chunk 拼接）
        for ref in candidates:
            full_text = mixed_index.dense.page_text(ref.doc_id, ref.page)
            if full_text:
                ref.text = full_text

        candidates = [ref for ref in candidates if ref.text.strip()]
        if not candidates:
            return []

        # 精确词（mortgage / 第 60 页的 60）出现在哪些页，就锁哪些页——
        # 选页模型只看摘要，容易凭语义猜错，这一步保证“查词”永远不靠猜。
        strong_terms = _strong_terms(question)
        if strong_terms:
            hit_pages = [
                ref
                for ref in candidates
                if any(term in ref.text.lower() for term in strong_terms)
            ]
            if hit_pages:
                return hit_pages[:MAX_PAGES]

        if len(candidates) <= MAX_PAGES:
            return candidates

        picked = await self._choose(question, candidates)
        if not picked:
            return candidates[:MAX_PAGES]
        # 无论如何都带上召回排名第一的页，避免模型漏选
        merged = [candidates[0]] + [page for page in picked if page is not candidates[0]]
        return merged[:MAX_PAGES]

    async def _choose(self, question: str, pages: List[PageRef]) -> List[PageRef]:
        terms = extract_terms(question)
        lines = []
        for index, page in enumerate(pages):
            hits = [term for term in terms if term and term in page.text.lower()][:3]
            marker = f"（命中：{'、'.join(hits)}）" if hits else ""
            lines.append(
                f"[{index + 1}] {page.source} 第 {page.page} 页{marker} —— "
                f"{' '.join(page.text.split())[:120]}"
            )
        listing = "\n".join(lines)
        prompt = f"""你要回答用户问题，先决定读哪几页原文。

用户问题：{question}

候选页面（编号、文档、页码、开头摘要）：
{listing}

规则：
- 最多选 {MAX_PAGES} 页，只能从候选里选；
- 优先选最可能包含具体答案的页（含数字、术语、定义的那几页）；
- 只输出 JSON：{{"pages": [1, 3]}}"""
        try:
            client = get_client()
            response = await client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
                response_format={"type": "json_object"},
            )
            payload = json.loads(response.choices[0].message.content or "{}")
            picks: List[int] = []
            for value in payload.get("pages", []):
                try:
                    picks.append(int(value))
                except (TypeError, ValueError):
                    continue
            chosen = [pages[pick - 1] for pick in picks if 1 <= pick <= len(pages)]
            return chosen[:MAX_PAGES]
        except Exception as exc:
            logger.warning("选页失败，按召回顺序取前 %d 页：%s", MAX_PAGES, exc)
            return []

    @staticmethod
    def sources(pages: List[PageRef], query: str = "") -> List[Dict]:
        return [
            {
                "index": index,
                "chunk_id": "",
                "doc_id": page.doc_id,
                "source": page.source,
                "page": page.page,
                "snippet": " ".join(page.text.split())[:160],
                "terms": matched_terms(query, page.text),
            }
            for index, page in enumerate(pages, start=1)
        ]


# 全局单例
precise_reader = PreciseReader()
