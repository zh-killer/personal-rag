"""PersonalRAG — FastAPI 入口

接口：
  POST /api/upload            上传并索引（PDF / Markdown / TXT）
  GET  /api/documents         文档列表
  DELETE /api/documents/{id}  删除文档及其索引
  POST /api/query             非流式问答
  POST /api/query/stream      流式问答（SSE，JSON 事件）
  GET  /api/health            健康检查
  GET  /api/stats             索引统计
  POST /api/cache/clear       清空 QA 缓存
如果 frontend/dist 存在，会直接托管前端页面，只需要启动这一个服务。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import time
import uuid

import fitz  # PyMuPDF
from starlette.background import BackgroundTask
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .retrieval.verify import summarize, verify_citations
from .cache.cache_manager import QACache
from .core.config import config
from .core.llm import LLMNotConfigured
from .indexing.indexer import mixed_index
from .indexing.layout import page_markdown
from .indexing.tokenizer import extract_terms
from .services.backup import conversation_markdown, conversations_markdown, create_backup
from .services.conversations import ConversationStore
from .services.tasks import TaskManager
from .services.usage import UsageStore
from .retrieval.generator import generator
from .retrieval.reranker import Reranker
from .retrieval.retriever import retriever
from .retrieval.lookup import try_vocab_answer
from .retrieval.precise import precise_reader
from .retrieval.rewriter import QueryRewriter
from .retrieval.router import QueryRouter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("personal-rag")

UPLOAD_TMP = Path(config.index.data_dir) / ".uploads"
_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".md": "text/markdown; charset=utf-8",
    ".markdown": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
}

router = QueryRouter()
rewriter = QueryRewriter()
reranker = Reranker(use_llm=True)
qa_cache = QACache()
tasks = TaskManager()
conversations = ConversationStore(Path(config.index.data_dir) / "conversations.db")
usage = UsageStore(Path(config.index.data_dir) / "usage.db")
_INGEST_LOCK = asyncio.Lock()
_BACKGROUND: set[asyncio.Task] = set()


# ─── 生命周期 ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_TMP.mkdir(parents=True, exist_ok=True)
    for warning in config.validate():
        logger.warning(warning)
    logger.info(
        "LLM=%s @ %s | Embedding=%s | 数据目录=%s",
        config.llm.model,
        config.llm.api_base,
        config.embedding.provider,
        config.index.data_dir,
    )
    try:
        stats = await asyncio.to_thread(mixed_index.warm_up)
        logger.info("索引就绪：%s", stats)
    except Exception:
        logger.exception("索引初始化失败，接口仍会启动，请检查 chroma_db 权限")
    yield


app = FastAPI(
    title="PersonalRAG",
    description="个人知识库 RAG 问答（PDF / Markdown / TXT）",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 请求 / 响应模型 ─────────────────────────────────────
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


# ─── 工具 ────────────────────────────────────────────────
def sse(event: dict) -> str:
    """SSE 帧：内容整体做 JSON 转义，换行不会破坏协议。"""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _display_filename(raw: Optional[str]) -> str:
    name = Path(str(raw or "").replace("\\", "/")).name.strip()
    return name or "未命名文档"


def _history_payload(turns: List[ChatTurn]) -> List[dict]:
    return [turn.model_dump() for turn in turns[-config.history_turns:]]


async def _prepare(
    question: str,
    history: List[dict],
    use_rerank: bool,
    top_k: int,
    doc_id: Optional[str] = None,
) -> tuple[List[dict], float]:
    decision = router.route(question, has_history=bool(history))
    variants = await rewriter.rewrite(question, history, use_llm=decision.use_rewrite)
    result = await retriever.retrieve(
        question, variants, top_k=top_k or decision.top_k, doc_id=doc_id
    )
    chunks = result.chunks
    if use_rerank and chunks:
        chunks = await reranker.rerank(question, chunks)
    return chunks, result.confidence


def _search_snippet(content: str, query: str, width: int = 160) -> str:
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


def _meta_answer() -> str:
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


# ─── 文档管理 ────────────────────────────────────────────
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """只负责落盘并创建后台任务，立即返回 task_id。"""
    filename = _display_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in config.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式：{suffix or '未知'}。支持：{', '.join(config.supported_formats)}",
        )

    UPLOAD_TMP.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOAD_TMP / f"{uuid.uuid4().hex}{suffix}"
    limit = config.max_upload_size_mb * 1024 * 1024
    size = 0
    try:
        with temp_path.open("wb") as out:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"文件超过 {config.max_upload_size_mb} MB 限制",
                    )
                out.write(block)
        if size == 0:
            raise HTTPException(status_code=400, detail="文件为空")
    except HTTPException:
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"保存文件失败：{exc}") from exc
    finally:
        await file.close()

    task = tasks.create(filename)
    job = asyncio.create_task(_run_ingest(task.id, temp_path, filename))
    _BACKGROUND.add(job)
    job.add_done_callback(_BACKGROUND.discard)

    return JSONResponse(
        status_code=202,
        content={
            "task_id": task.id,
            "status": "queued",
            "filename": filename,
            "size_bytes": size,
        },
    )


async def _run_ingest(task_id: str, temp_path: Path, filename: str) -> None:
    """后台入库：串行执行，避免同时占满 CPU 与向量库写入。"""
    async with _INGEST_LOCK:
        tasks.update(task_id, status="running", stage="开始解析", progress=0.02)
        try:
            result = await asyncio.to_thread(_ingest_sync, task_id, temp_path, filename)
            tasks.update(
                task_id,
                status="done",
                stage="完成",
                progress=1.0,
                doc_id=result.doc_id,
                chunk_count=result.chunk_count,
                message="内容未变化，已跳过重建" if result.skipped else "已建立索引",
            )
        except Exception as exc:
            logger.exception("异步入库失败：%s", filename)
            temp_path.unlink(missing_ok=True)
            tasks.update(
                task_id,
                status="error",
                stage="失败",
                progress=1.0,
                error=str(exc) or exc.__class__.__name__,
                message="入库失败",
            )


def _ingest_sync(task_id: str, temp_path: Path, filename: str):
    def progress(stage: str, fraction: float) -> None:
        tasks.update(task_id, stage=stage, progress=max(0.0, min(1.0, fraction)))

    return mixed_index.add_file(temp_path, filename, progress=progress)


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return task


@app.get("/api/tasks")
async def list_tasks(limit: int = 20):
    return {"tasks": tasks.list(limit)}


# ─── 对话历史 ────────────────────────────────────────────
@app.get("/api/conversations")
async def list_conversations(limit: int = 50, q: str = ""):
    return {
        "conversations": await asyncio.to_thread(conversations.list, limit, q)
    }


@app.post("/api/conversations")
async def create_conversation(body: ConversationCreate):
    return await asyncio.to_thread(conversations.create, body.title)


@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, body: ConversationRename):
    ok = await asyncio.to_thread(conversations.rename, conversation_id, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True, "title": body.title.strip()[:60]}


@app.get("/api/conversations/{conversation_id}/export")
async def export_conversation(conversation_id: str):
    content = await asyncio.to_thread(
        conversation_markdown, conversations, conversation_id
    )
    if not content:
        raise HTTPException(status_code=404, detail="对话不存在")
    filename = f"personalrag-chat-{conversation_id}.md"
    return PlainTextResponse(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.post("/api/conversations/{conversation_id}/export/save")
async def save_conversation_export(conversation_id: str):
    """把导出内容写到本机 backend/exports/，返回绝对路径。"""
    content = await asyncio.to_thread(
        conversation_markdown, conversations, conversation_id
    )
    if not content:
        raise HTTPException(status_code=404, detail="对话不存在")

    export_dir = Path(config.index.data_dir).parent / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    filename = f"personalrag-chat-{time.strftime('%Y%m%d-%H%M%S')}.md"
    path = export_dir / filename

    try:
        await asyncio.to_thread(path.write_text, content, "utf-8")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"写入失败：{exc}") from exc

    return {
        "success": True,
        "filename": filename,
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
    }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    data = await asyncio.to_thread(conversations.get, conversation_id)
    if not data:
        raise HTTPException(status_code=404, detail="对话不存在")
    return data


@app.post("/api/conversations/{conversation_id}/messages")
async def append_message(conversation_id: str, body: MessageAppend):
    message_id = await asyncio.to_thread(
        conversations.append,
        conversation_id,
        body.role,
        body.content,
        body.sources,
        body.confidence,
    )
    if message_id is None:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True, "id": message_id}


@app.post("/api/conversations/{conversation_id}/messages/delete")
async def delete_conversation_messages(conversation_id: str, body: MessageIds):
    removed = await asyncio.to_thread(
        conversations.delete_messages, conversation_id, body.ids
    )
    return {"success": True, "removed": removed}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    ok = await asyncio.to_thread(conversations.delete, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True}


@app.get("/api/documents")
async def list_documents():
    docs = await asyncio.to_thread(mixed_index.list_docs)
    return {"documents": docs}


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not re.fullmatch(r"[A-Za-z0-9_-]{4,64}", doc_id):
        raise HTTPException(status_code=400, detail="非法的文档 ID")
    record = await asyncio.to_thread(mixed_index.remove_file, doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="文档不存在或已被删除")

    stored = Path(str(record.get("path", "")))
    data_dir = Path(config.index.data_dir).resolve()
    try:
        if stored.is_file() and data_dir in stored.resolve().parents:
            stored.unlink()
    except OSError as exc:
        logger.warning("删除源文件失败：%s", exc)
    return {"success": True, "doc_id": doc_id, "filename": record.get("source", "")}


# ─── 原文检索（不调用大模型）──────────────────────────────
@app.get("/api/documents/{doc_id}/file")
async def document_file(doc_id: str):
    """返回原始文件（inline），用于预览 / 跳到指定页。"""
    record = mixed_index.store.get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="文档不存在")

    path = Path(str(record.get("path", "")))
    data_dir = Path(config.index.data_dir).resolve()
    try:
        resolved = path.resolve()
        if not resolved.is_file() or data_dir not in resolved.parents:
            raise HTTPException(status_code=404, detail="源文件不存在")
    except OSError as exc:
        raise HTTPException(status_code=404, detail="源文件不存在") from exc

    media_type = _MEDIA_TYPES.get(resolved.suffix.lower(), "application/octet-stream")
    return FileResponse(
        resolved,
        media_type=media_type,
        filename=str(record.get("source", resolved.name)),
        content_disposition_type="inline",
    )


@app.get("/api/documents/{doc_id}/page/{page}/image")
async def document_page_image(doc_id: str, page: int, terms: str = ""):
    """把 PDF 的某一页渲染成 PNG，并把查询词用黄色标出来。"""
    record = mixed_index.store.get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="文档不存在")

    path = Path(str(record.get("path", "")))
    if path.suffix.lower() != ".pdf" or not path.is_file():
        raise HTTPException(status_code=400, detail="只有 PDF 支持页面渲染")

    term_list = [item.strip() for item in terms.split(",") if item.strip()][:8]

    try:
        png = await asyncio.to_thread(_render_page_png, path, page, term_list)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("页面渲染失败：%s", exc)
        raise HTTPException(status_code=500, detail=f"页面渲染失败：{exc}") from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _render_page_png(path: Path, page: int, terms: list[str] | None = None) -> bytes:
    document = fitz.open(str(path))
    try:
        if page < 1 or page > document.page_count:
            raise ValueError("页码超出范围")
        pdf_page = document[page - 1]

        for term in terms or []:
            if len(term) > 40:
                continue
            try:
                rects = pdf_page.search_for(term)
            except Exception:
                continue
            for rect in rects:
                try:
                    annot = pdf_page.add_highlight_annot(rect)
                    annot.set_colors(stroke=(1.0, 0.86, 0.2))
                    annot.update()
                except Exception:
                    continue

        pixmap = pdf_page.get_pixmap(matrix=fitz.Matrix(2, 2))
        return pixmap.tobytes("png")
    finally:
        document.close()


@app.get("/api/documents/{doc_id}/page/{page}")
async def document_page(doc_id: str, page: int):
    """某页的完整文本 + 版面重建后的 Markdown（用于原文预览）。"""
    record = mixed_index.store.get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="文档不存在")

    path = Path(str(record.get("path", "")))
    text = await asyncio.to_thread(mixed_index.dense.page_text, doc_id, page)

    suffix = path.suffix.lower()
    markdown = ""
    if suffix == ".pdf" and path.is_file():
        kind = "pdf"
        try:
            markdown = await asyncio.to_thread(page_markdown, str(path), page)
        except Exception as exc:
            logger.warning("版面重建失败：%s", exc)
    elif suffix in (".md", ".markdown") and path.is_file():
        kind = "markdown"
        try:
            markdown = await asyncio.to_thread(path.read_text, "utf-8", "ignore")
        except Exception:
            markdown = ""
    elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        kind = "image"
    else:
        kind = "text"

    if not markdown.strip() and not text.strip():
        raise HTTPException(status_code=404, detail="该页没有可显示的文本")

    return {
        "doc_id": doc_id,
        "source": record.get("source", ""),
        "page": page,
        "text": text,
        "markdown": markdown,
        "kind": kind,
    }


@app.post("/api/search")
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
                "snippet": _search_snippet(content, body.query),
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
                    "snippet": _search_snippet(content, body.query),
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


# ─── 问答 ────────────────────────────────────────────────
@app.post("/api/query", response_model=QueryResponse)
async def query(body: QueryRequest):
    started = time.perf_counter()
    history = _history_payload(body.history)
    decision = router.route(body.question, has_history=bool(history))
    mode = body.mode if body.mode != "auto" else router.pick_mode(body.question)

    if decision.query_type == "meta":
        return QueryResponse(
            answer=_meta_answer(),
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

    chunks, confidence = await _prepare(
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


@app.post("/api/query/stream")
async def query_stream(body: QueryRequest, request: Request):
    async def event_stream() -> AsyncIterator[str]:
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            history = _history_payload(body.history)
            decision = router.route(body.question, has_history=bool(history))
            mode = body.mode if body.mode != "auto" else router.pick_mode(body.question)

            if decision.query_type == "meta":
                yield sse({"type": "sources", "sources": [], "confidence": 1.0})
                yield sse({"type": "token", "text": _meta_answer()})
                yield sse({"type": "done", "confidence": 1.0, "elapsed_ms": elapsed(), "cached": False})
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
                        {"type": "done", "confidence": 0.0, "elapsed_ms": elapsed(), "cached": False}
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
                    {"type": "done", "confidence": cached.get("confidence", 0.0), "elapsed_ms": elapsed(), "cached": True}
                )
                return

            chunks, confidence = await _prepare(
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
                yield sse({"type": "done", "confidence": 0.0, "elapsed_ms": elapsed(), "cached": False})
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


# ─── 系统 ────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": app.version,
        "llm_configured": config.llm.configured,
        "model": config.llm.model,
        "embedding": mixed_index.dense.description,
        "documents": len(mixed_index.store.all()),
        "chunks": mixed_index.dense.count(),
        "fts_ready": mixed_index.fts.ready,
        "vocab_entries": mixed_index.vocab.count(),
    }


@app.get("/api/stats")
async def stats():
    data = await asyncio.to_thread(mixed_index.stats)
    data.update(
        {
            "model": config.llm.model,
            "embedding_provider": config.embedding.provider,
            "chunk_size": config.chunk.chunk_size,
            "top_k": config.retrieval.top_k,
            "supported_formats": list(config.supported_formats),
            "max_upload_size_mb": config.max_upload_size_mb,
        }
    )
    return data


@app.post("/api/cache/clear")
async def clear_cache():
    return {"success": True, "removed": qa_cache.clear()}


# ─── 用量与导出 ──────────────────────────────────────────
@app.get("/api/usage/summary")
async def usage_summary(days: int = 7):
    return await asyncio.to_thread(usage.summary, days)


@app.get("/api/export/backup")
async def export_backup():
    """打包原始文档 + 对话记录，供备份/迁移。"""
    try:
        zip_path, temp_dir = await asyncio.to_thread(
            create_backup,
            Path(config.index.data_dir),
            Path(config.index.data_dir) / "conversations.db",
            {"documents_indexed": len(mixed_index.store.all())},
        )
    except Exception as exc:
        logger.exception("备份失败")
        raise HTTPException(status_code=500, detail=f"备份失败：{exc}") from exc

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_path.name,
        background=BackgroundTask(shutil.rmtree, temp_dir, True),
    )


@app.get("/api/export/conversations")
async def export_conversations():
    """把所有对话导出成 Markdown。"""
    content = await asyncio.to_thread(conversations_markdown, conversations)
    filename = f"personalrag-conversations-{time.strftime('%Y%m%d-%H%M')}.md"
    return PlainTextResponse(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"服务器内部错误：{exc}"})


# ─── 前端静态资源（构建后只需启动后端）───────────────────
DIST = Path(config.frontend_dist)
if DIST.is_dir():
    assets = DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        icon = DIST / "favicon.svg"
        return FileResponse(icon) if icon.is_file() else JSONResponse(status_code=404, content={})

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path.startswith("api/") or full_path in {"docs", "redoc", "openapi.json"}:
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (DIST / full_path).resolve()
        try:
            candidate.relative_to(DIST.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Not Found")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")
else:

    @app.get("/", include_in_schema=False)
    async def root_hint():
        return JSONResponse(
            {
                "message": "PersonalRAG API 已启动，但未找到前端构建产物。",
                "api_docs": "/docs",
                "hint": "在 frontend 目录执行 npm install && npm run build，然后重启后端。",
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=config.reload)
