"""多路召回 + RRF 融合

旧版把「语义相似度」和「BM25 原始分」直接加权相加，量纲不同、还会随改写查询数量累加，
导致同一个 chunk 的分数能超过 1，低置信阈值形同虚设。
这里统一用 RRF（Reciprocal Rank Fusion）：只看排名，天然可比。
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.config import config
from ..indexing.indexer import mixed_index
from ..indexing.tokenizer import extract_terms

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    top_similarity: float = 0.0


class MultiRetriever:
    def __init__(self) -> None:
        self.bm25_weight = min(max(config.retrieval.bm25_weight, 0.0), 1.0)
        self.dense_weight = 1.0 - self.bm25_weight

    async def retrieve(
        self,
        query: str,
        variants: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        doc_id: Optional[str] = None,
    ) -> RetrievalResult:
        variants = variants or [query]
        return await asyncio.to_thread(
            self._retrieve_sync, variants, top_k or config.retrieval.top_k, doc_id
        )

    # ── 同步实现（embedding / BM25 都是 CPU 或阻塞调用，放到线程池里跑）──
    def _retrieve_sync(
        self, variants: List[str], top_k: int, doc_id: Optional[str] = None
    ) -> RetrievalResult:
        fetch_k = max(config.retrieval.fetch_k, top_k * 3)
        rrf_k = max(1, config.retrieval.rrf_k)
        where = {"doc_id": doc_id} if doc_id else None
        scores: Dict[str, Dict] = {}

        for variant in variants:
            try:
                dense_hits = mixed_index.dense.search(variant, fetch_k, where=where)
            except Exception as exc:
                logger.warning("向量检索失败：%s", exc)
                dense_hits = []

            for rank, hit in enumerate(dense_hits):
                entry = scores.setdefault(
                    hit["chunk_id"],
                    {
                        "chunk_id": hit["chunk_id"],
                        "content": hit.get("content", ""),
                        "metadata": hit.get("metadata", {}),
                        "retrieval_score": 0.0,
                        "dense_sim": 0.0,
                        "bm25_score": 0.0,
                    },
                )
                entry["retrieval_score"] += self.dense_weight / (rrf_k + rank + 1)
                entry["dense_sim"] = max(entry["dense_sim"], float(hit.get("dense_sim", 0.0)))

            # 限定文档时先过滤再计数，避免 FTS 名额被其他文档占满
            bm25_rank = 0
            for hit in mixed_index.fts.search(
                variant, fetch_k * 2 if doc_id else fetch_k
            ):
                if doc_id and hit["doc_id"] != doc_id:
                    continue
                entry = scores.setdefault(
                    hit["chunk_id"],
                    {
                        "chunk_id": hit["chunk_id"],
                        "content": hit["content"],
                        "metadata": {
                            "doc_id": hit["doc_id"],
                            "source": hit["source"],
                            "page": hit["page"],
                            "chunk_idx": hit["chunk_idx"],
                        },
                        "retrieval_score": 0.0,
                        "dense_sim": 0.0,
                        "bm25_score": 0.0,
                    },
                )
                entry["retrieval_score"] += self.bm25_weight / (rrf_k + bm25_rank + 1)
                entry["bm25_score"] = max(entry["bm25_score"], float(hit["score"]))
                bm25_rank += 1
                if bm25_rank >= fetch_k:
                    break

        self._apply_exact_bonus(variants[0], scores)

        ranked = sorted(
            scores.values(), key=lambda item: item["retrieval_score"], reverse=True
        )[:top_k]

        confidence, top_similarity = self._confidence(ranked, bool(mixed_index.fts.ready))
        return RetrievalResult(ranked, confidence, top_similarity)

    @staticmethod
    def _apply_exact_bonus(query: str, scores: Dict[str, Dict]) -> None:
        """精确词加成：mortgage / RAG 这类词只出现在个别片段里时，
        RRF 的排名分压不过一堆语义近邻，需要单独加权。"""
        exact_terms = {
            term for term in extract_terms(query) if term.isascii() and len(term) >= 3
        }
        if not exact_terms:
            return
        for entry in scores.values():
            content = str(entry.get("content", "")).lower()
            hits = sum(1 for term in exact_terms if term in content)
            if hits:
                entry["retrieval_score"] += min(hits, 3) * config.retrieval.exact_bonus
                entry["exact_hits"] = hits

    @staticmethod
    def _confidence(chunks: List[Dict], bm25_ready: bool) -> tuple[float, float]:
        """用前 3 个片段的 dense 相似度均值映射到 0~1，BM25 命中时给一个保底分。"""
        sims = sorted(
            (float(c.get("dense_sim") or 0.0) for c in chunks[:3]),
            reverse=True,
        )
        if sims and sims[0] > 0:
            top = sims[0]
            mean = sum(sims) / len(sims)
            span = max(1e-6, config.retrieval.confidence_span)
            confidence = (mean - config.retrieval.confidence_floor) / span
            return round(min(1.0, max(0.0, confidence)), 3), round(top, 3)
        if chunks and bm25_ready:
            return 0.25, 0.0
        return 0.0, 0.0


# 全局单例
retriever = MultiRetriever()
