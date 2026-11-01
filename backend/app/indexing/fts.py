"""SQLite FTS5 关键词索引（替代内存 BM25）

相比原来的 BM25Okapi：
- 持久化：重启不用重建，几十份文档也能秒开；
- 增量：上传只插入本文件的片段，删除按 doc_id 精确清理；
- 中文：写入前把正文切成「单字 + 二元组」token 串，查询用同一套 tokenizer；
- 排序：直接用 FTS5 内置 bm25()。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Iterable, List

from .splitter import Chunk
from .tokenizer import extract_terms, tokenize

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    doc_id UNINDEXED,
    source UNINDEXED,
    page UNINDEXED,
    chunk_idx UNINDEXED,
    content UNINDEXED,
    tokens,
    tokenize='unicode61'
)
"""


class FTSIndex:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    @property
    def ready(self) -> bool:
        return True

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])

    def add(self, chunks: Iterable[Chunk]) -> None:
        rows = [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source,
                int(chunk.page or 0),
                int(chunk.metadata.get("chunk_idx", 0) or 0),
                chunk.content,
                " ".join(tokenize(chunk.content)),
            )
            for chunk in chunks
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany(
                "INSERT INTO chunks_fts "
                "(chunk_id, doc_id, source, page, chunk_idx, content, tokens) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def remove_doc(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            self._conn.commit()

    def replace_doc(self, doc_id: str, chunks: Iterable[Chunk]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
            self._conn.commit()
            self.add(chunks)

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chunks_fts")
            self._conn.commit()

    def search(self, query: str, top_k: int = 10, doc_id: str | None = None) -> List[Dict]:
        tokens = extract_terms(query) or tokenize(query)
        unique: List[str] = []
        for token in tokens:
            token = token.strip()
            if token and token not in unique:
                unique.append(token)
        if not unique:
            return []

        match = " OR ".join('"' + token.replace('"', '""') + '"' for token in unique[:32])
        sql = (
            "SELECT chunk_id, doc_id, source, page, chunk_idx, content, bm25(chunks_fts) AS score "
            "FROM chunks_fts WHERE chunks_fts MATCH ?"
        )
        params: list = [match]
        if doc_id:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY score LIMIT ?"
        params.append(int(top_k))
        try:
            with self._lock:
                rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            logger.warning("FTS 查询失败：%s", exc)
            return []

        return [
            {
                "chunk_id": row[0],
                "doc_id": row[1],
                "source": row[2],
                "page": int(row[3] or 0),
                "chunk_idx": int(row[4] or 0),
                "content": row[5] or "",
                # bm25() 越小越相关，取反后与向量分数同向
                "score": -float(row[6]),
            }
            for row in rows
        ]
