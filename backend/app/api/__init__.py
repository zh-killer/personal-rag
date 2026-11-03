"""API 路由包。

每个模块负责一组端点，main.py 只做 include_router。
"""
from __future__ import annotations

from .conversations import router as conversations_router
from .documents import router as documents_router
from .query import router as query_router
from .system import router as system_router

routers = (documents_router, query_router, conversations_router, system_router)

__all__ = ["routers"]
