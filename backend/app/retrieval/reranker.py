"""LLM 批量重排

旧版对每条 chunk 单独发一次请求（最多 20 次串行，首 token 前就要等十几秒）。
这里改成「一次请求给所有候选打分」，延迟从 N 次降到 1 次。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, List, Optional

from ..core.config import config
from ..core.llm import get_client

logger = logging.getLogger(__name__)


class Reranker:
    PROMPT = """你是检索结果重排器。判断每个候选片段对回答用户问题有多大帮助。

用户问题：{query}

候选片段：
{candidates}

评分标准：
- 90-100：直接包含答案或关键事实
- 60-89：包含相关背景，需要推理
- 20-59：主题相关但无法回答
- 0-19：无关

只输出 JSON：{{"scores": [{{"id": 1, "score": 92}}, {{"id": 2, "score": 41}}]}}"""

    def __init__(self, use_llm: bool = True) -> None:
        self.use_llm = use_llm

    async def rerank(
        self, query: str, chunks: List[Dict], top_k: Optional[int] = None
    ) -> List[Dict]:
        top_k = top_k or config.retrieval.rerank_top_k
        if not chunks:
            return []
        if not self.use_llm or not config.llm.configured:
            return chunks[:top_k]

        candidates = chunks[: max(top_k, config.retrieval.rerank_max_chunks)]
        blocks = [
            f"[{index + 1}] {chunk.get('content', '')[:600]}"
            for index, chunk in enumerate(candidates)
        ]
        prompt = self.PROMPT.format(query=query, candidates="\n\n".join(blocks))

        try:
            client = get_client()
            response = await client.chat.completions.create(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            scored = self._parse(response.choices[0].message.content or "")
        except Exception as exc:
            logger.warning("重排失败，保持融合排序：%s", exc)
            return chunks[:top_k]

        for index, chunk in enumerate(candidates):
            chunk["rerank_score"] = scored.get(index + 1)
            chunk["final_score"] = (
                float(chunk["rerank_score"]) if chunk.get("rerank_score") is not None else -1.0
            )

        ranked = sorted(
            candidates,
            key=lambda c: (c.get("final_score", -1.0), c.get("retrieval_score", 0.0)),
            reverse=True,
        )

        # 过滤明显无关的片段，避免把噪声塞进上下文
        selected = [c for c in ranked if (c.get("rerank_score") or 0) >= 35]
        if not selected:
            selected = ranked[:2]
        return selected[:top_k]

    @staticmethod
    def _parse(text: str) -> Dict[int, float]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
        scores: Dict[int, float] = {}
        for item in payload.get("scores", []):
            try:
                scores[int(item["id"])] = float(item["score"])
            except (KeyError, TypeError, ValueError):
                continue
        return scores
