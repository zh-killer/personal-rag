"""问答用量统计（估算）

只做本地估算，不追求与账单完全一致：
- 记录每次问答的模式、问题/答案长度、估算 token、模型调用次数、耗时
- 按天汇总，供「文档」页展示
"""
from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List


def estimate_tokens(question: str, answer: str) -> int:
    """中文按约 1.6 字符/token 粗估。"""
    return max(1, int((len(question) + len(answer)) / 1.6))


class UsageStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                mode TEXT NOT NULL,
                question_chars INTEGER NOT NULL,
                answer_chars INTEGER NOT NULL,
                est_tokens INTEGER NOT NULL,
                calls INTEGER NOT NULL,
                elapsed_ms INTEGER NOT NULL
            )
            """
        )
        self._conn.commit()

    def record(
        self,
        mode: str,
        question: str,
        answer: str,
        calls: int,
        elapsed_ms: int,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage (ts, mode, question_chars, answer_chars, est_tokens, calls, elapsed_ms) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    time.time(),
                    mode,
                    len(question),
                    len(answer),
                    estimate_tokens(question, answer),
                    max(0, calls),
                    max(0, elapsed_ms),
                ),
            )
            self._conn.commit()

    def summary(self, days: int = 7) -> Dict:
        days = max(1, min(365, days))
        cutoff = time.time() - days * 86400
        with self._lock:
            total = self._conn.execute(
                "SELECT count(*), coalesce(sum(est_tokens), 0), coalesce(sum(calls), 0), "
                "coalesce(avg(elapsed_ms), 0) FROM usage WHERE ts >= ?",
                (cutoff,),
            ).fetchone()
            by_mode = self._conn.execute(
                "SELECT mode, count(*), coalesce(sum(est_tokens), 0), coalesce(sum(calls), 0) "
                "FROM usage WHERE ts >= ? GROUP BY mode ORDER BY count(*) DESC",
                (cutoff,),
            ).fetchall()
            daily = self._conn.execute(
                "SELECT date(ts, 'unixepoch', 'localtime') AS day, count(*), "
                "coalesce(sum(est_tokens), 0), coalesce(sum(calls), 0) "
                "FROM usage WHERE ts >= ? GROUP BY day ORDER BY day DESC LIMIT ?",
                (cutoff, days),
            ).fetchall()

        return {
            "days": days,
            "questions": int(total[0]),
            "estimated_tokens": int(total[1]),
            "model_calls": int(total[2]),
            "avg_elapsed_ms": int(total[3]),
            "by_mode": [
                {
                    "mode": row[0],
                    "questions": int(row[1]),
                    "estimated_tokens": int(row[2]),
                    "model_calls": int(row[3]),
                }
                for row in by_mode
            ],
            "daily": [
                {
                    "day": row[0],
                    "questions": int(row[1]),
                    "estimated_tokens": int(row[2]),
                    "model_calls": int(row[3]),
                }
                for row in daily
            ],
        }

    def recent(self, limit: int = 20) -> List[Dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, mode, est_tokens, calls, elapsed_ms FROM usage "
                "ORDER BY id DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        return [
            {
                "timestamp": row[0],
                "mode": row[1],
                "estimated_tokens": int(row[2]),
                "model_calls": int(row[3]),
                "elapsed_ms": int(row[4]),
            }
            for row in rows
        ]
