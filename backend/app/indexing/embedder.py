"""文本向量化

三种模式（EMBEDDING_PROVIDER）：
- local：sentence-transformers 本地模型（默认，离线可用）
- api  ：OpenAI 兼容 /embeddings 接口
- auto ：先试 api，失败回落 local

注意：同一个知识库必须始终使用同一种 embedding，否则向量维度/语义空间不一致。
切换前请清空 backend/chroma_db（或让程序自动迁移）。
"""
from __future__ import annotations

import os
import threading
from typing import List

from ..core.config import config


class EmbeddingError(RuntimeError):
    """向量化失败（带可读原因）"""


class Embedder:
    def __init__(self) -> None:
        self.provider = config.embedding.provider
        self._local_model = None
        self._api_client = None
        self._mode: str | None = None
        self._lock = threading.Lock()

    # ── 本地模型 ─────────────────────────────────────────
    @property
    def local_model(self):
        if self._local_model is None:
            with self._lock:
                if self._local_model is None:
                    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise EmbeddingError(
                            "未安装 sentence-transformers。请执行："
                            "pip install sentence-transformers（或改用 EMBEDDING_PROVIDER=api）"
                        ) from exc
                    try:
                        self._local_model = SentenceTransformer(config.embedding.local_model)
                    except Exception as exc:
                        raise EmbeddingError(
                            f"加载本地向量模型失败（{config.embedding.local_model}）：{exc}。"
                            "首次运行需要联网下载，可设置 HF_ENDPOINT 镜像。"
                        ) from exc
        return self._local_model

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        model = self.local_model
        with self._lock:  # SentenceTransformer 不保证线程安全
            vectors = model.encode(
                texts,
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vectors]

    # ── API ──────────────────────────────────────────────
    @property
    def api_client(self):
        if self._api_client is None:
            from openai import OpenAI

            if not config.embedding.api_key:
                raise EmbeddingError("EMBEDDING_PROVIDER=api 但未配置 API Key")
            self._api_client = OpenAI(
                api_key=config.embedding.api_key,
                base_url=config.embedding.api_base,
                timeout=config.embedding.timeout,
            )
        return self._api_client

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        batch_size = config.embedding.batch_size
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            try:
                response = self.api_client.embeddings.create(
                    model=config.embedding.model, input=batch
                )
            except Exception as exc:
                raise EmbeddingError(f"Embedding API 调用失败：{exc}") from exc
            vectors.extend(item.embedding for item in response.data)
        return vectors

    # ── 对外接口 ──────────────────────────────────────────
    def _resolve_mode(self) -> str:
        if self._mode:
            return self._mode
        if self.provider == "api":
            self._mode = "api"
        elif self.provider == "local":
            self._mode = "local"
        else:  # auto
            try:
                self._embed_api(["ping"])
                self._mode = "api"
            except Exception:
                self._mode = "local"
        return self._mode

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        mode = self._resolve_mode()
        return self._embed_api(texts) if mode == "api" else self._embed_local(texts)

    def embed_one(self, text: str) -> List[float]:
        vectors = self.embed([text])
        if not vectors:
            raise EmbeddingError("向量化返回空结果")
        return vectors[0]

    @property
    def description(self) -> str:
        if self.provider == "local":
            return f"local:{config.embedding.local_model}"
        return f"{self.provider}:{config.embedding.model}"
