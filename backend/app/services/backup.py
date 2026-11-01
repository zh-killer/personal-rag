"""知识库备份与对话导出

备份内容：
- documents/：上传的原始文件（data/ 下的文档）
- conversations.db：对话历史（用 SQLite backup API 拷贝，运行中也安全）
- manifest.json / README.txt：清单与恢复说明

不打包向量库与 FTS/词条库：它们可以从原始文档重建，
避免把运行中的 Chroma 二进制文件拷成半成品。
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SKIP_SUFFIXES = {".db", ".db-wal", ".db-shm", ".tmp"}


def _copy_sqlite(source: Path, target: Path) -> None:
    """用 SQLite 在线备份 API 拷贝，避免直接复制 WAL 造成不一致。"""
    src = sqlite3.connect(str(source))
    try:
        dst = sqlite3.connect(str(target))
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def create_backup(
    data_dir: Path,
    conversations_db: Path,
    extra: Optional[Dict] = None,
) -> Tuple[Path, Path]:
    """生成备份 zip，返回 (zip 路径, 临时目录)。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    temp_dir = Path(tempfile.mkdtemp(prefix="personalrag-backup-"))
    zip_path = temp_dir / f"personalrag-backup-{stamp}.zip"

    documents: List[str] = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue
        documents.append(path.name)

    manifest = {
        "app": "PersonalRAG",
        "version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "documents": documents,
        "conversations_db": conversations_db.exists(),
    }
    if extra:
        manifest.update(extra)

    readme = (
        "PersonalRAG 备份\n"
        "================\n\n"
        "恢复方法：\n"
        "1. 停止服务（关闭 start.bat 窗口或 Ctrl+C）\n"
        "2. 把 documents/ 里的文件复制回 backend/data/\n"
        "3. 把 conversations.db 复制回 backend/data/\n"
        "4. 重新启动服务：会自动解析文档并重建向量 / 关键词 / 词条索引\n\n"
        "注：索引未包含在备份中，恢复后需要重新索引；扫描件会重新 OCR。\n"
    )

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(data_dir.iterdir()):
            if path.name not in documents:
                continue
            archive.write(path, f"documents/{path.name}")

        if conversations_db.exists():
            temp_db = temp_dir / "conversations.db"
            _copy_sqlite(conversations_db, temp_db)
            archive.write(temp_db, "conversations.db")

        archive.writestr(
            "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2)
        )
        archive.writestr("README.txt", readme)

    return zip_path, temp_dir


def conversation_markdown(store, conversation_id: str) -> str:
    """把单个对话导出成 Markdown。"""
    detail = store.get(conversation_id)
    if not detail:
        return ""

    lines = [
        f"# {detail['title'] or '未命名对话'}",
        "",
        f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 消息数：{len(detail['messages'])}",
        "",
    ]
    for message in detail["messages"]:
        role = "用户" if message["role"] == "user" else "助手"
        lines.append(f"**{role}：** {message['content']}")
        for source in message.get("sources") or []:
            page = f" 第 {source.get('page')} 页" if source.get("page") else ""
            lines.append(f"> 来源：{source.get('source', '')}{page}")
        lines.append("")

    return "\n".join(lines)


def conversations_markdown(store) -> str:
    """把所有对话导出成一份可读的 Markdown。"""
    conversations = store.list(limit=500)
    lines = [
        "# PersonalRAG 对话导出",
        "",
        f"> 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"> 共 {len(conversations)} 个对话",
        "",
    ]

    for summary in conversations:
        detail = store.get(summary["id"])
        if not detail:
            continue
        updated = datetime.fromtimestamp(detail["updated_at"]).strftime("%Y-%m-%d %H:%M")
        lines.append(f"## {detail['title'] or '未命名对话'}")
        lines.append(f"*最后更新：{updated}*")
        lines.append("")
        for message in detail["messages"]:
            role = "用户" if message["role"] == "user" else "助手"
            lines.append(f"**{role}：** {message['content']}")
            for source in message.get("sources") or []:
                page = f" 第 {source.get('page')} 页" if source.get("page") else ""
                lines.append(f"> 来源：{source.get('source', '')}{page}")
            lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
