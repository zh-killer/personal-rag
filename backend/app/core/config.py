"""
PersonalRAG 全局配置

所有可调参数集中在这里，支持通过 backend/.env 覆盖。
导入时只读环境变量、不初始化任何外部客户端，保证「启动失败」时能给出可诊断的信息。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent.parent   # backend/
PROJECT_DIR = BACKEND_DIR.parent                               # 项目根目录


def _load_dotenv() -> None:
    """加载 .env；没装 python-dotenv 时静默跳过（不阻断启动）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for candidate in (BACKEND_DIR / ".env", PROJECT_DIR / ".env"):
        if candidate.is_file():
            load_dotenv(candidate, override=False)


_load_dotenv()


def _str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _int(name: str, default: int) -> int:
    try:
        return int(_str(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_str(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    return _str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


@dataclass
class LLMConfig:
    """LLM 配置（OpenAI 兼容协议：DeepSeek / 硅基流动 / OpenAI 均可）"""
    model: str = field(default_factory=lambda: _str("LLM_MODEL", "deepseek-chat"))
    api_key: str = field(
        default_factory=lambda: _str("OPENAI_API_KEY", "") or _str("DEEPSEEK_API_KEY", "")
    )
    api_base: str = field(
        default_factory=lambda: _str("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    )
    temperature: float = field(default_factory=lambda: _float("LLM_TEMPERATURE", 0.1))
    max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 2048))
    timeout: float = field(default_factory=lambda: _float("LLM_TIMEOUT", 90.0))

    @property
    def configured(self) -> bool:
        return bool(self.api_key) and not self.api_key.startswith("sk-your")


@dataclass
class EmbeddingConfig:
    """
    Embedding 配置。

    provider:
      - local ：只用 sentence-transformers 本地模型（默认，离线可用、无 API 费用）
      - api   ：只用 OpenAI 兼容的 /embeddings 接口
      - auto  ：先试 api，失败后回落到本地模型
    """
    provider: str = field(default_factory=lambda: _str("EMBEDDING_PROVIDER", "local").lower())
    model: str = field(default_factory=lambda: _str("EMBEDDING_MODEL", "text-embedding-3-small"))
    api_key: str = field(
        default_factory=lambda: _str("EMBEDDING_API_KEY", "") or _str("OPENAI_API_KEY", "")
    )
    api_base: str = field(
        default_factory=lambda: _str("EMBEDDING_API_BASE", "")
        or _str("OPENAI_API_BASE", "https://api.deepseek.com/v1")
    )
    local_model: str = field(
        default_factory=lambda: _str("EMBEDDING_LOCAL_MODEL", "BAAI/bge-small-zh-v1.5")
    )
    batch_size: int = field(default_factory=lambda: _int("EMBEDDING_BATCH_SIZE", 64))
    timeout: float = field(default_factory=lambda: _float("EMBEDDING_TIMEOUT", 60.0))


@dataclass
class ChunkConfig:
    """文档切分配置"""
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 60))
    min_chunk_chars: int = field(default_factory=lambda: _int("MIN_CHUNK_CHARS", 20))
    separators: list = field(
        default_factory=lambda: [
            "\n\n", "\n", "。", "！", "？", ".", "!", "?", "；", ";", "，", ",", " ", "",
        ]
    )


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = field(default_factory=lambda: _int("TOP_K", 10))
    fetch_k: int = field(default_factory=lambda: _int("FETCH_K", 40))
    rerank_top_k: int = field(default_factory=lambda: _int("RERANK_TOP_K", 6))
    rerank_max_chunks: int = field(default_factory=lambda: _int("RERANK_MAX_CHUNKS", 20))
    bm25_weight: float = field(default_factory=lambda: _float("BM25_WEIGHT", 0.45))
    rrf_k: int = field(default_factory=lambda: _int("RRF_K", 60))
    # 查询词在片段中原样出现时的加成（解决「mortgage 明明在库里却检索不到」）
    exact_bonus: float = field(default_factory=lambda: _float("EXACT_BONUS", 0.03))
    # 置信度映射：dense_sim 在 [floor, floor+span] 上线性映射到 [0, 1]
    confidence_floor: float = field(default_factory=lambda: _float("CONFIDENCE_FLOOR", 0.35))
    confidence_span: float = field(default_factory=lambda: _float("CONFIDENCE_SPAN", 0.35))


@dataclass
class IndexConfig:
    """索引配置"""
    persist_dir: str = field(
        default_factory=lambda: _str("CHROMA_DIR", str(BACKEND_DIR / "chroma_db"))
    )
    collection_name: str = field(
        default_factory=lambda: _str("COLLECTION_NAME", "personal_knowledge")
    )
    data_dir: str = field(default_factory=lambda: _str("DATA_DIR", str(BACKEND_DIR / "data")))
    # 检测到旧版索引（chunk 没有 doc_id 元数据）时自动重建
    auto_migrate: bool = field(default_factory=lambda: _bool("AUTO_MIGRATE", True))


@dataclass
class CacheConfig:
    """QA 结果缓存（按 问题 + 知识库版本 命中，索引一变自动失效）"""
    enable_qa_cache: bool = field(default_factory=lambda: _bool("ENABLE_QA_CACHE", True))
    cache_dir: str = field(default_factory=lambda: _str("CACHE_DIR", str(BACKEND_DIR / "cache")))
    qa_cache_ttl: int = field(default_factory=lambda: _int("QA_CACHE_TTL", 3600))


@dataclass
class AppConfig:
    """应用全局配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)

    host: str = field(default_factory=lambda: _str("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _int("PORT", 8000))
    reload: bool = field(default_factory=lambda: _bool("RELOAD", False))
    max_upload_size_mb: int = field(default_factory=lambda: _int("MAX_UPLOAD_MB", 50))
    history_turns: int = field(default_factory=lambda: _int("HISTORY_TURNS", 5))
    ocr_enabled: bool = field(default_factory=lambda: _bool("OCR_ENABLED", True))
    supported_formats: tuple = (
        ".pdf", ".md", ".markdown", ".txt",
        ".png", ".jpg", ".jpeg", ".webp", ".bmp",
    )
    cors_origins: tuple = ("http://localhost:5173", "http://127.0.0.1:5173")
    frontend_dist: Path = field(default_factory=lambda: PROJECT_DIR / "frontend" / "dist")

    def validate(self) -> list[str]:
        """启动自检，返回需要提示给用户的告警（不抛异常）。"""
        warnings: list[str] = []
        if not self.llm.configured:
            warnings.append(
                "未检测到可用的 LLM API Key：请在 backend/.env 设置 OPENAI_API_KEY，"
                "并确认 OPENAI_API_BASE / LLM_MODEL 正确。聊天接口会返回 503。"
            )
        if self.embedding.provider == "local":
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                warnings.append(
                    "EMBEDDING_PROVIDER=local 但未安装 sentence-transformers，"
                    "请执行：pip install sentence-transformers"
                )
        if not self.frontend_dist.is_dir():
            warnings.append(
                f"未找到前端构建产物 {self.frontend_dist}，后端只提供 API。"
                "请在前端目录执行 npm install && npm run build，或使用 npm run dev 开发模式。"
            )
        return warnings


# 全局单例
config = AppConfig()
