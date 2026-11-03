"""PersonalRAG — FastAPI 入口（只做装配）

路由按职责拆到 app/api/：
  documents.py      上传 / 入库任务 / 文档台账 / 原文与页面渲染
  query.py          关键词检索 / 问答 / SSE 流式问答
  conversations.py  对话历史的增删改查与导出
  system.py         健康检查 / 统计 / 缓存 / 用量 / 备份

如果 frontend/dist 存在，会直接托管前端页面，只需要启动这一个服务。
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import routers
from .core.config import config
from .deps import UPLOAD_TMP, mixed_index

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
logger = logging.getLogger("personal-rag")


# ─── 生命周期 ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_TMP.mkdir(parents=True, exist_ok=True)
    for warning in config.validate():
        logger.warning(warning)
    logger.info(
        "LLM=%s @ %s | Embedding=%s | 数据目录=%s",
        config.llm.model,
        config.llm.api_base,
        config.embedding.provider,
        config.index.data_dir,
    )
    try:
        stats = await asyncio.to_thread(mixed_index.warm_up)
        logger.info("索引就绪：%s", stats)
    except Exception:
        logger.exception("索引初始化失败，接口仍会启动，请检查 chroma_db 权限")
    yield


app = FastAPI(
    title="PersonalRAG",
    description="个人知识库 RAG 问答（PDF / Markdown / TXT）",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(config.cors_origins),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 未处理异常：统一返回 JSON，避免前端拿到 HTML 堆栈 ───
@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    logger.exception("未处理异常：%s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": f"服务器内部错误：{exc}"})


# ─── 业务路由 ────────────────────────────────────────────
for _router in routers:
    app.include_router(_router)


# ─── 前端静态资源（构建后只需启动后端）───────────────────
DIST = Path(config.frontend_dist)
if DIST.is_dir():
    assets = DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        icon = DIST / "favicon.svg"
        return FileResponse(icon) if icon.is_file() else JSONResponse(status_code=404, content={})

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(full_path: str):
        if full_path.startswith("api/") or full_path in {"docs", "redoc", "openapi.json"}:
            raise HTTPException(status_code=404, detail="Not Found")
        candidate = (DIST / full_path).resolve()
        try:
            candidate.relative_to(DIST.resolve())
        except ValueError:
            raise HTTPException(status_code=404, detail="Not Found")
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")

else:

    @app.get("/", include_in_schema=False)
    async def root_hint():
        return JSONResponse(
            {
                "message": "PersonalRAG API 已启动，但未找到前端构建产物。",
                "api_docs": "/docs",
                "hint": "在 frontend 目录执行 npm install && npm run build，然后重启后端。",
            }
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=config.host, port=config.port, reload=config.reload)
