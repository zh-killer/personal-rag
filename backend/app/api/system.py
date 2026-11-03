"""系统信息与导出：健康检查 / 统计 / 缓存清理 / 用量 / 备份。"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse
from starlette.background import BackgroundTask

from ..core.config import config
from ..deps import conversations, mixed_index, qa_cache, usage
from ..services.backup import conversations_markdown, create_backup

logger = logging.getLogger("personal-rag.system")

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/health")
async def health(request: Request):
    return {
        "status": "ok",
        "version": request.app.version,
        "llm_configured": config.llm.configured,
        "model": config.llm.model,
        "embedding": mixed_index.dense.description,
        "documents": len(mixed_index.store.all()),
        "chunks": mixed_index.dense.count(),
        "fts_ready": mixed_index.fts.ready,
        "vocab_entries": mixed_index.vocab.count(),
    }


@router.get("/stats")
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


@router.post("/cache/clear")
async def clear_cache():
    return {"success": True, "removed": qa_cache.clear()}


@router.get("/usage/summary")
async def usage_summary(days: int = 7):
    return await asyncio.to_thread(usage.summary, days)


@router.get("/export/backup")
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


@router.get("/export/conversations")
async def export_conversations():
    """把所有对话导出成 Markdown。"""
    content = await asyncio.to_thread(conversations_markdown, conversations)
    filename = f"personalrag-conversations-{time.strftime('%Y%m%d-%H%M')}.md"
    return PlainTextResponse(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


__all__ = ["router"]
