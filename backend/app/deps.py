"""进程内共享的单例与服务。

原先这些对象都堆在 main.py 顶部，路由一多就分不清「谁在用谁」。
集中到这里之后：路由模块只依赖 deps + 自己的业务函数，main.py 只负责装配。
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from .cache.cache_manager import QACache
from .core.config import config
from .indexing.indexer import mixed_index
from .retrieval.generator import generator
from .retrieval.precise import precise_reader
from .retrieval.reranker import Reranker
from .retrieval.retriever import retriever
from .retrieval.rewriter import QueryRewriter
from .retrieval.router import QueryRouter
from .services.conversations import ConversationStore
from .services.tasks import TaskManager
from .services.usage import UsageStore

# 上传临时目录（入库成功后由索引层移动到 data/）
UPLOAD_TMP = Path(config.index.data_dir) / ".uploads"

# 问答链路上的无状态组件
query_router = QueryRouter()
rewriter = QueryRewriter()
reranker = Reranker(use_llm=True)

# 有状态的服务
qa_cache = QACache()
tasks = TaskManager()
conversations = ConversationStore(Path(config.index.data_dir) / "conversations.db")
usage = UsageStore(Path(config.index.data_dir) / "usage.db")

# 入库串行化：同时解析多个大文件会把 CPU 和向量库写打满
ingest_lock = asyncio.Lock()
background_tasks: set[asyncio.Task] = set()

__all__ = [
    "UPLOAD_TMP",
    "background_tasks",
    "conversations",
    "generator",
    "ingest_lock",
    "mixed_index",
    "precise_reader",
    "qa_cache",
    "query_router",
    "reranker",
    "retriever",
    "rewriter",
    "tasks",
    "usage",
]
