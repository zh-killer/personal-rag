"""对话记录：SQLite 持久化，支持多会话列表与回看。

之前对话只存在浏览器 localStorage 的「当前一个会话」里，
刷新能恢复，但一按「新对话」就永久丢失，也看不到更早的记录。
这里把会话与消息落到服务端，刷新、换标签页、隔几天回来都能翻。
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    confidence REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


class ConversationStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def create(self, title: str = "") -> Dict:
        now = time.time()
        conversation_id = uuid.uuid4().hex[:12]
        title = (title or "新对话").strip()[:60]
        with self._lock:
            self._conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conversation_id, title, now, now),
            )
            self._conn.commit()
        return {"id": conversation_id, "title": title, "created_at": now, "updated_at": now}

    def list(self, limit: int = 50, query: str = "") -> List[Dict]:
        query = (query or "").strip()
        with self._lock:
            if query:
                like = f"%{query}%"
                rows = self._conn.execute(
                    "SELECT c.id, c.title, c.created_at, c.updated_at, "
                    "(SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) "
                    "FROM conversations c "
                    "WHERE c.title LIKE ? OR EXISTS ("
                    "  SELECT 1 FROM messages m WHERE m.conversation_id = c.id AND m.content LIKE ?"
                    ") ORDER BY c.updated_at DESC LIMIT ?",
                    (like, like, max(1, limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT c.id, c.title, c.created_at, c.updated_at, "
                    "(SELECT count(*) FROM messages m WHERE m.conversation_id = c.id) "
                    "FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
                    (max(1, limit),),
                ).fetchall()
        return [
            {
                "id": row[0],
                "title": row[1],
                "created_at": row[2],
                "updated_at": row[3],
                "message_count": int(row[4]),
            }
            for row in rows
        ]

    def get(self, conversation_id: str) -> Optional[Dict]:
        with self._lock:
            conversation = self._conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
            if not conversation:
                return None
            rows = self._conn.execute(
                "SELECT id, role, content, sources, confidence, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()

        messages = []
        for row in rows:
            try:
                sources = json.loads(row[3] or "[]")
            except json.JSONDecodeError:
                sources = []
            messages.append(
                {
                    "id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "sources": sources,
                    "confidence": row[4],
                    "created_at": row[5],
                }
            )
        return {
            "id": conversation[0],
            "title": conversation[1],
            "created_at": conversation[2],
            "updated_at": conversation[3],
            "messages": messages,
        }

    def append(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        confidence: Optional[float] = None,
    ) -> Optional[int]:
        now = time.time()
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                return None
            cursor = self._conn.execute(
                "INSERT INTO messages "
                "(conversation_id, role, content, sources, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    conversation_id,
                    role,
                    content,
                    json.dumps(sources or [], ensure_ascii=False),
                    confidence,
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def rename(self, conversation_id: str, title: str) -> bool:
        title = (title or "").strip()[:60]
        if not title:
            return False
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, time.time(), conversation_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def delete_messages(self, conversation_id: str, message_ids: List[int]) -> int:
        ids = [int(item) for item in message_ids][:100]
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            cursor = self._conn.execute(
                f"DELETE FROM messages WHERE conversation_id = ? AND id IN ({placeholders})",
                [conversation_id, *ids],
            )
            self._conn.commit()
            return cursor.rowcount

    def delete(self, conversation_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            self._conn.execute(
                "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM conversations").fetchone()[0])
