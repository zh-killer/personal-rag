"""问答与原文检索。

/api/search   关键词精确检索（不调用大模型）
/api/query    非流式问答
/api/query/stream  SSE 流式问答
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..core.config import config
from ..core.llm import LLMNotConfigured
from ..deps import generator, mixed_index, precise_reader, qa_cache, query_router, retriever, usage
from ..indexing.tokenizer import extract_terms
from ..retrieval.lookup import try_vocab_answer
from ..retrieval.verify import summarize, verify_citations
from ..schemas import QueryRequest, QueryResponse, SearchRequest
from .common import history_payload, meta_answer, prepare_chunks, search_snippet, sse

logger = logging.getLogger("personal-rag.query")

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/search")
async def search_chunks(body: SearchRequest):
    """关键词检索：只有真正包含关键词的片段才算结果；没有就明确说没有。"""
    if body.doc_id:
        known = {doc["doc_id"] for doc in mixed_index.list_docs()}
        if body.doc_id not in known:
            raise HTTPException(status_code=404, detail="文档不存在")

    terms = [term for term in extract_terms(body.query) if term]
    if not terms:
        terms = [body.query.strip().lower()]
    terms = list(dict.fromkeys(terms))

    candidates = await asyncio.to_thread(
        mixed_index.fts.search, body.query, body.top_k * 4, body.doc_id
    )

    exact: list[dict] = []
    seen: set[str] = set()
    for candidate in candidates:
        content = str(candidate.get("content", ""))
        content_lower = content.lower()
        matched = [term for term in terms if term and term in content_lower]
        if not matched or candidate["chunk_id"] in seen:
            continue
        seen.add(candidate["chunk_id"])
        exact.append(
            {
                "chunk_id": candidate["chunk_id"],
                "doc_id": candidate["doc_id"],
                "source": candidate["source"],
                "page": candidate["page"],
                "snippet": search_snippet(content, body.query),
                "score": round(float(candidate.get("score", 0)), 4),
                "matched_terms": matched[:5],
                "match": "exact",
            }
        )

    exact.sort(key=lambda hit: (-len(hit["matched_terms"]), -hit["score"]))
    hits = exact[: body.top_k]

    # 一个关键词都没命中时，才给出「语义相关」片段，并且单独标注
    related: list[dict] = []
    if not hits:
        result = await retriever.retrieve(
            body.query, [body.query], top_k=3, doc_id=body.doc_id
        )
        for chunk in result.chunks:
            metadata = chunk.get("metadata") or {}
            content = str(chunk.get("content", ""))
            related.append(
                {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "doc_id": metadata.get("doc_id", ""),
                    "source": metadata.get("source", ""),
                    "page": metadata.get("page"),
                    "snippet": search_snippet(content, body.query),
                    "score": round(float(chunk.get("retrieval_score", 0)), 4),
                    "matched_terms": [],
                    "match": "semantic",
                }
            )

    return {
        "query": body.query,
        "doc_id": body.doc_id,
        "count": len(hits),
        "hits": hits,
        "related": related,
        "terms": terms[:5],
    }


@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    started = time.perf_counter()
    history = history_payload(body.history)
    decision = query_router.route(body.question, has_history=bool(history))
    mode = body.mode if body.mode != "auto" else query_router.pick_mode(body.question)

    if decision.query_type == "meta":
        return QueryResponse(
            answer=meta_answer(),
            sources=[],
            confidence=1.0,
            retrieval_count=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    # 词表类文档：命中词条就直接回答，不花一次大模型调用
    lookup = try_vocab_answer(body.question, mixed_index.vocab, body.doc_id)
    if lookup:
        return QueryResponse(
            answer=lookup["answer"],
            sources=lookup["sources"],
            confidence=1.0,
            retrieval_count=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            cached=False,
        )

    if not config.llm.configured:
        raise HTTPException(status_code=503, detail="未配置 LLM API Key，无法生成回答")

    if mode == "precise":
        pages = await precise_reader.select_pages(body.question, body.doc_id)
        if not pages:
            return QueryResponse(
                answer="这几份文档里没有找到相关页面。换个说法，或在「搜索」页直接搜关键词。",
                sources=[],
                confidence=0.0,
                retrieval_count=0,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
        answer = await generator.answer_pages(
            body.question, [page.as_dict() for page in pages], history
        )
        checks = verify_citations(answer, [{"content": page.text} for page in pages])
        usage.record(
            "precise",
            body.question,
            answer,
            2,
            int((time.perf_counter() - started) * 1000),
        )
        return QueryResponse(
            answer=answer,
            sources=precise_reader.sources(pages, body.question),
            confidence=0.85,
            retrieval_count=len(pages),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            checks=checks,
        )

    version = mixed_index.doc_version()
    # 多轮对话的答案依赖上下文，不能按「问题」复用缓存
    use_cache = not history
    cached = qa_cache.get(body.question, version, body.doc_id or "") if use_cache else None
    if cached:
        return QueryResponse(
            answer=cached["answer"],
            sources=cached.get("sources", []),
            confidence=cached.get("confidence", 0.0),
            retrieval_count=cached.get("retrieval_count", 0),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
            cached=True,
        )

    chunks, confidence = await prepare_chunks(
        body.question, history, body.use_rerank, decision.top_k, body.doc_id
    )
    if not chunks:
        return QueryResponse(
            answer="知识库里没有检索到与这个问题相关的内容。可以换个说法，或先上传相关文档。",
            sources=[],
            confidence=0.0,
            retrieval_count=0,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    answer = await generator.answer(body.question, chunks, history)
    sources = generator.sources(chunks, body.question)
    checks = verify_citations(answer, chunks)
    if use_cache:
        qa_cache.set(
            body.question,
            version,
            {
                "answer": answer,
                "sources": sources,
                "confidence": confidence,
                "retrieval_count": len(chunks),
            },
            body.doc_id or "",
        )
    usage.record(
        "fast",
        body.question,
        answer,
        2 if body.use_rerank else 1,
        int((time.perf_counter() - started) * 1000),
    )
    return QueryResponse(
        answer=answer,
        sources=sources,
        confidence=confidence,
        retrieval_count=len(chunks),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        checks=checks,
    )


@router.post("/query/stream")
async def query_stream(body: QueryRequest, request: Request):
    async def event_stream() -> AsyncIterator[str]:
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            history = history_payload(body.history)
            decision = query_router.route(body.question, has_history=bool(history))
            mode = body.mode if body.mode != "auto" else query_router.pick_mode(body.question)

            if decision.query_type == "meta":
                yield sse({"type": "sources", "sources": [], "confidence": 1.0})
                yield sse({"type": "token", "text": meta_answer()})
                yield sse(
                    {"type": "done", "confidence": 1.0, "elapsed_ms": elapsed(), "cached": False}
                )
                return

            lookup = try_vocab_answer(body.question, mixed_index.vocab, body.doc_id)
            if lookup:
                yield sse(
                    {
                        "type": "sources",
                        "sources": lookup["sources"],
                        "confidence": 1.0,
                    }
                )
                yield sse({"type": "token", "text": lookup["answer"]})
                yield sse(
                    {"type": "done", "confidence": 1.0, "elapsed_ms": elapsed(), "cached": False}
                )
                return

            if not config.llm.configured:
                raise LLMNotConfigured(
                    "未配置 LLM API Key：请在 backend/.env 设置 OPENAI_API_KEY 后重启服务"
                )

            if mode == "precise":
                pages = await precise_reader.select_pages(body.question, body.doc_id)
                sources = precise_reader.sources(pages, body.question)
                precision_confidence = 0.85 if pages else 0.0
                yield sse(
                    {
                        "type": "sources",
                        "sources": sources,
                        "confidence": precision_confidence,
                    }
                )
                if not pages:
                    yield sse(
                        {
                            "type": "token",
                            "text": "这几份文档里没有找到相关页面。换个说法，或在「搜索」页直接搜关键词。",
                        }
                    )
                    yield sse(
                        {
                            "type": "done",
                            "confidence": 0.0,
                            "elapsed_ms": elapsed(),
                            "cached": False,
                        }
                    )
                    return
                page_parts: List[str] = []
                async for token in generator.stream_pages(
                    body.question, [page.as_dict() for page in pages], history
                ):
                    if await request.is_disconnected():
                        logger.info("客户端已断开，停止生成")
                        return
                    page_parts.append(token)
                    yield sse({"type": "token", "text": token})

                page_answer = "".join(page_parts)
                page_checks = verify_citations(
                    page_answer, [{"content": page.text} for page in pages]
                )
                usage.record("precise", body.question, page_answer, 2, elapsed())
                yield sse(
                    {
                        "type": "verify",
                        "checks": page_checks,
                        "unverified": summarize(page_checks)["unverified"],
                    }
                )
                yield sse(
                    {
                        "type": "done",
                        "confidence": precision_confidence,
                        "elapsed_ms": elapsed(),
                        "cached": False,
                    }
                )
                return

            version = mixed_index.doc_version()
            use_cache = not history
            cached = qa_cache.get(body.question, version, body.doc_id or "") if use_cache else None
            if cached:
                yield sse(
                    {
                        "type": "sources",
                        "sources": cached.get("sources", []),
                        "confidence": cached.get("confidence", 0.0),
                    }
                )
                yield sse({"type": "token", "text": cached["answer"]})
                yield sse(
                    {
                        "type": "done",
                        "confidence": cached.get("confidence", 0.0),
                        "elapsed_ms": elapsed(),
                        "cached": True,
                    }
                )
                return

            chunks, confidence = await prepare_chunks(
                body.question, history, body.use_rerank, decision.top_k, body.doc_id
            )
            sources = generator.sources(chunks, body.question) if chunks else []
            yield sse({"type": "sources", "sources": sources, "confidence": confidence})

            if not chunks:
                yield sse(
                    {
                        "type": "token",
                        "text": "知识库里没有检索到与这个问题相关的内容。可以换个说法，或先上传相关文档。",
                    }
                )
                yield sse(
                    {"type": "done", "confidence": 0.0, "elapsed_ms": elapsed(), "cached": False}
                )
                return

            parts: List[str] = []
            async for token in generator.stream(body.question, chunks, history):
                if await request.is_disconnected():
                    logger.info("客户端已断开，停止生成")
                    return
                parts.append(token)
                yield sse({"type": "token", "text": token})

            answer = "".join(parts)
            checks = verify_citations(answer, chunks)
            usage.record(
                "fast",
                body.question,
                answer,
                2 if body.use_rerank else 1,
                elapsed(),
            )
            yield sse(
                {
                    "type": "verify",
                    "checks": checks,
                    "unverified": summarize(checks)["unverified"],
                }
            )
            if use_cache:
                qa_cache.set(
                    body.question,
                    version,
                    {
                        "answer": answer,
                        "sources": sources,
                        "confidence": confidence,
                        "retrieval_count": len(chunks),
                    },
                    body.doc_id or "",
                )
            yield sse(
                {"type": "done", "confidence": confidence, "elapsed_ms": elapsed(), "cached": False}
            )
        except asyncio.CancelledError:
            raise
        except LLMNotConfigured as exc:
            yield sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # 把异常变成前端可读的事件，而不是断流
            logger.exception("问答失败")
            yield sse({"type": "error", "message": f"生成失败：{exc}"})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
