"""文档入库与原文查看。

负责：上传落盘 → 后台入库任务 → 文档台账 / 删除 → 原始文件 / 单页渲染。
"""
from __future__ import annotations

import asyncio
import logging
import re
import uuid
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ..core.config import config
from ..deps import UPLOAD_TMP, background_tasks, ingest_lock, mixed_index, tasks
from ..indexing.layout import page_markdown

logger = logging.getLogger("personal-rag.documents")

router = APIRouter(prefix="/api", tags=["documents"])

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


def display_filename(raw: Optional[str]) -> str:
    name = Path(str(raw or "").replace("\\", "/")).name.strip()
    return name or "未命名文档"


@router.post("/upload", status_code=202)
async def upload_document(file: UploadFile = File(...)):
    """只负责落盘并创建后台任务，立即返回 task_id。"""
    filename = display_filename(file.filename)
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
    job = asyncio.create_task(run_ingest(task.id, temp_path, filename))
    background_tasks.add(job)
    job.add_done_callback(background_tasks.discard)

    return JSONResponse(
        status_code=202,
        content={
            "task_id": task.id,
            "status": "queued",
            "filename": filename,
            "size_bytes": size,
        },
    )


async def run_ingest(task_id: str, temp_path: Path, filename: str) -> None:
    """后台入库：串行执行，避免同时占满 CPU 与向量库写入。"""
    async with ingest_lock:
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


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return task


@router.get("/tasks")
async def list_tasks(limit: int = 20):
    return {"tasks": tasks.list(limit)}


@router.get("/documents")
async def list_documents():
    docs = await asyncio.to_thread(mixed_index.list_docs)
    return {"documents": docs}


@router.delete("/documents/{doc_id}")
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


@router.get("/documents/{doc_id}/file")
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


@router.get("/documents/{doc_id}/page/{page}/image")
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
        png = await asyncio.to_thread(render_page_png, path, page, term_list)
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


def render_page_png(path: Path, page: int, terms: list[str] | None = None) -> bytes:
    """把 PDF 指定页渲染成 PNG，可选高亮命中词。"""
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


@router.get("/documents/{doc_id}/page/{page}")
async def document_page(doc_id: str, page: int):
    """某页的完整文本 + 版面重建后的 Markdown（用于原文预览）。"""
    record = mixed_index.store.get(doc_id)
    if not record:
        raise HTTPException(status_code=404, detail="文档不存在")

    path = Path(str(record.get("path", "")))
    text = await asyncio.to_thread(mixed_index.dense.page_text, doc_id, page)

    suffix = path.suffix.lower()
    markdown = ""
    kind = "text"
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


__all__ = ["router", "render_page_png", "run_ingest"]
