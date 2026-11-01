"""词表类文档的结构化抽取与存储

很多资料是「编号 + 单词 + 音标 + 词性 + 释义」的词表（四六级、考研、术语表）。
这类内容最适合入库时抽成表：之后查词是数据库精确命中，不靠向量猜。

抽取是纯规则的：不调用大模型，又快又稳定。
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_CJK = re.compile(r"[\u4e00-\u9fff]")
# 词条边界：行内连续出现 "12. word" 时也能切开
_SPLIT = re.compile(r"(?<![\dA-Za-z])(\d{1,3})[.、]\s*(?=[A-Za-z])")
_WORD = re.compile(r"^([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,2})")
_PHONETIC = re.compile(r"\[([^\]]{1,48})\]")
_POS = re.compile(
    r"\b(n|v|vt|vi|adj|adv|a|ad|prep|conj|pron|num|art|int|aux|abbr)\.",
    re.IGNORECASE,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vocab (
    word TEXT NOT NULL,
    word_lower TEXT NOT NULL,
    phonetic TEXT DEFAULT '',
    pos TEXT DEFAULT '',
    meaning TEXT NOT NULL,
    page INTEGER DEFAULT 0,
    doc_id TEXT NOT NULL,
    source TEXT DEFAULT '',
    seq INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_vocab_word ON vocab(word_lower);
CREATE INDEX IF NOT EXISTS idx_vocab_doc ON vocab(doc_id);
"""


@dataclass
class VocabEntry:
    word: str
    meaning: str
    page: int
    doc_id: str
    source: str
    phonetic: str = ""
    pos: str = ""
    seq: int = 0


def extract_entries(
    pages: Sequence[Tuple[int, str]], doc_id: str, source: str
) -> List[VocabEntry]:
    """从按页文本里抽出词条；抽不到就返回空列表（调用方据此判断文档类型）。"""
    entries: List[VocabEntry] = []
    seq = 0

    for page, text in pages:
        flattened = " ".join(text.split())
        matches = list(_SPLIT.finditer(flattened))
        for index, match in enumerate(matches):
            start = match.end()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(flattened)
            content = flattened[start:end].strip()
            if not content:
                continue

            cjk = _CJK.search(content)
            if not cjk or cjk.start() == 0:
                continue
            head = content[: cjk.start()]
            meaning = content[cjk.start() :].strip()
            if not meaning:
                continue

            word_match = _WORD.match(head)
            if not word_match:
                continue
            word = word_match.group(1).strip()
            if len(word) < 2:
                continue

            tail = head[word_match.end() :]
            phonetic_match = _PHONETIC.search(tail)
            pos_values = [item.group(1).lower() for item in _POS.finditer(tail)]

            entries.append(
                VocabEntry(
                    word=word,
                    meaning=meaning,
                    page=page,
                    doc_id=doc_id,
                    source=source,
                    phonetic=phonetic_match.group(1).strip() if phonetic_match else "",
                    pos=" / ".join(dict.fromkeys(pos_values)),
                    seq=seq,
                )
            )
            seq += 1

    return entries


class VocabStore:
    """词条表（SQLite），按 doc_id 整体替换/删除。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def replace_doc(self, doc_id: str, entries: Iterable[VocabEntry]) -> int:
        rows = [
            (
                entry.word,
                entry.word.lower(),
                entry.phonetic,
                entry.pos,
                entry.meaning,
                int(entry.page or 0),
                entry.doc_id,
                entry.source,
                int(entry.seq),
            )
            for entry in entries
        ]
        with self._lock:
            self._conn.execute("DELETE FROM vocab WHERE doc_id = ?", (doc_id,))
            if rows:
                self._conn.executemany(
                    "INSERT INTO vocab "
                    "(word, word_lower, phonetic, pos, meaning, page, doc_id, source, seq) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            self._conn.commit()
        return len(rows)

    def remove_doc(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM vocab WHERE doc_id = ?", (doc_id,))
            self._conn.commit()

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM vocab")
            self._conn.commit()

    def count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT count(*) FROM vocab").fetchone()[0])

    def count_doc(self, doc_id: str) -> int:
        with self._lock:
            return int(
                self._conn.execute(
                    "SELECT count(*) FROM vocab WHERE doc_id = ?", (doc_id,)
                ).fetchone()[0]
            )

    def lookup(self, word: str, doc_id: str | None = None) -> List[Dict]:
        sql = (
            "SELECT word, phonetic, pos, meaning, page, doc_id, source "
            "FROM vocab WHERE word_lower = ?"
        )
        params: list = [word.strip().lower()]
        if doc_id:
            sql += " AND doc_id = ?"
            params.append(doc_id)
        sql += " ORDER BY page, seq LIMIT 8"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "word": row[0],
                "phonetic": row[1],
                "pos": row[2],
                "meaning": row[3],
                "page": row[4],
                "doc_id": row[5],
                "source": row[6],
            }
            for row in rows
        ]

    def stats(self) -> Dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*), count(DISTINCT doc_id) FROM vocab"
            ).fetchone()
        return {"entries": int(row[0]), "documents": int(row[1])}
