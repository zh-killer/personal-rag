"""对话历史：列表 / 新建 / 改名 / 追加消息 / 导出 / 删除。

会话数据存在 SQLite（backend/data/conversations.db），所以这里每个端点都
走 asyncio.to_thread，避免同步 IO 阻塞事件循环。
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from ..core.config import config
from ..deps import conversations
from ..schemas import ConversationCreate, ConversationRename, MessageAppend, MessageIds
from ..services.backup import conversation_markdown

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations(limit: int = 50, q: str = ""):
    return {"conversations": await asyncio.to_thread(conversations.list, limit, q)}


@router.post("")
async def create_conversation(body: ConversationCreate):
    return await asyncio.to_thread(conversations.create, body.title)


@router.get("/{conversation_id}/export")
async def export_conversation(conversation_id: str):
    content = await asyncio.to_thread(conversation_markdown, conversations, conversation_id)
    if not content:
        raise HTTPException(status_code=404, detail="对话不存在")
    filename = f"personalrag-chat-{conversation_id}.md"
    return PlainTextResponse(
        content,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/{conversation_id}/export/save")
async def save_conversation_export(conversation_id: str):
    """把导出内容写到本机 backend/exports/，返回绝对路径。"""
    content = await asyncio.to_thread(conversation_markdown, conversations, conversation_id)
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


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    data = await asyncio.to_thread(conversations.get, conversation_id)
    if not data:
        raise HTTPException(status_code=404, detail="对话不存在")
    return data


@router.patch("/{conversation_id}")
async def rename_conversation(conversation_id: str, body: ConversationRename):
    ok = await asyncio.to_thread(conversations.rename, conversation_id, body.title)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True, "title": body.title.strip()[:60]}


@router.post("/{conversation_id}/messages")
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


@router.post("/{conversation_id}/messages/delete")
async def delete_conversation_messages(conversation_id: str, body: MessageIds):
    removed = await asyncio.to_thread(conversations.delete_messages, conversation_id, body.ids)
    return {"success": True, "removed": removed}


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    ok = await asyncio.to_thread(conversations.delete, conversation_id)
    if not ok:
        raise HTTPException(status_code=404, detail="对话不存在")
    return {"success": True}


__all__ = ["router"]
