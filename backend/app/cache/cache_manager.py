"""QA 结果缓存

key = 问题 + 知识库版本 + 模型名：
- 上传/删除文档后 doc_version 变化，旧缓存自然失效（不会答出过时内容）；
- 命中后把整段答案当作一个 token 事件发回去，前端体验一致但零延迟。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

from ..core.config import config

logger = logging.getLogger(__name__)


class QACache:
    def __init__(self) -> None:
        self.dir = Path(config.cache.cache_dir) / "qa"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = max(60, config.cache.qa_cache_ttl)
        self.enabled = config.cache.enable_qa_cache

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def _key(self, question: str, doc_version: str, scope: str = "") -> str:
        raw = f"{question.strip().lower()}|{doc_version}|{config.llm.model}|{scope}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def get(self, question: str, doc_version: str, scope: str = "") -> Optional[Dict]:
        if not self.enabled:
            return None
        path = self._path(self._key(question, doc_version, scope))
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None
        if time.time() - float(data.get("timestamp", 0)) > self.ttl:
            path.unlink(missing_ok=True)
            return None
        return data.get("result")

    def set(self, question: str, doc_version: str, result: Dict, scope: str = "") -> None:
        if not self.enabled:
            return
        payload = {"timestamp": time.time(), "question": question, "result": result}
        path = self._path(self._key(question, doc_version, scope))
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning("写缓存失败：%s", exc)

    def clear(self) -> int:
        removed = 0
        for path in self.dir.glob("*.json"):
            try:
                path.unlink()
                removed += 1
            except OSError:
                pass
        return removed
