"""异步入库任务表

上传接口只负责落盘并创建任务，解析/OCR/向量化在后台跑，
前端轮询 /api/tasks/{id} 拿进度。任务存在进程内存里，重启即清空。
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskInfo:
    id: str
    filename: str
    status: str = "queued"  # queued | running | done | error
    stage: str = "排队中"
    progress: float = 0.0
    message: str = ""
    doc_id: str = ""
    chunk_count: int = 0
    error: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class TaskManager:
    def __init__(self, keep: int = 50) -> None:
        self._tasks: Dict[str, TaskInfo] = {}
        self._lock = threading.RLock()
        self._keep = max(5, keep)

    def create(self, filename: str) -> TaskInfo:
        with self._lock:
            self._prune()
            task = TaskInfo(id=uuid.uuid4().hex[:12], filename=filename)
            self._tasks[task.id] = task
            return task

    def update(self, task_id: str, **changes) -> Optional[TaskInfo]:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            for key, value in changes.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            task.updated_at = time.time()
            return task

    def get(self, task_id: str) -> Optional[dict]:
        with self._lock:
            task = self._tasks.get(task_id)
            return asdict(task) if task else None

    def list(self, limit: int = 20) -> List[dict]:
        with self._lock:
            items = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
            return [asdict(task) for task in items[: max(1, limit)]]

    def _prune(self) -> None:
        if len(self._tasks) < self._keep:
            return
        ordered = sorted(self._tasks.values(), key=lambda t: t.created_at)
        for task in ordered[: len(self._tasks) - self._keep + 1]:
            self._tasks.pop(task.id, None)
