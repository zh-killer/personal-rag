"""混合索引：ChromaDB 稠密向量 + BM25 稀疏关键词 + 文档台账

关键设计（修复旧版的核心问题）：
- doc_id 是稳定标识（上传时生成，不随内容变化），chunk 元数据里同时存 doc_id 和 source；
- 删除 / 重建都按 doc_id 精确匹配，不会留下孤儿向量；
- 文档台账 documents.json 记录 source / file_hash / path / chunk_count，便于增量与清理；
- 启动时从 Chroma 重建 BM25（旧版只在内存里，重启后关键词检索直接失效）；
- 检测到旧版索引（chunk 没有 doc_id）时自动从 data/ 重建一次。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings

from ..core.config import config
from .embedder import Embedder
from .fts import FTSIndex
from .parser import DocumentParser, ParsedDocument
from .splitter import Chunk, DocumentSplitter
from .naming import display_name, original_name, safe_filename
from .vocab import VocabStore, extract_entries

logger = logging.getLogger(__name__)

_COLLECTION_BATCH = 64


@dataclass
class IndexResult:
    doc_id: str
    filename: str
    chunk_count: int
    skipped: bool = False


# ─────────────────────────────────────────────────────────
# Chroma
# ─────────────────────────────────────────────────────────
class DenseIndex:
    def __init__(self) -> None:
        os.makedirs(config.index.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=config.index.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=config.index.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embedder = Embedder()

    def count(self) -> int:
        return int(self.collection.count())

    def add(
        self, chunks: List[Chunk], on_progress: Callable[[int, int], None] | None = None
    ) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed([c.content for c in chunks])
        total = len(chunks)
        for start in range(0, total, _COLLECTION_BATCH):
            batch = chunks[start : start + _COLLECTION_BATCH]
            try:
                self.collection.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=vectors[start : start + len(batch)],
                    documents=[c.content for c in batch],
                    metadatas=[c.metadata for c in batch],
                )
            except Exception as exc:
                raise ValueError(
                    f"写入向量库失败：{exc}。如果更换过 embedding 模型，"
                    "请删除 backend/chroma_db 后重新上传文档。"
                ) from exc
            if on_progress:
                on_progress(min(start + len(batch), total), total)

    def search(self, query: str, top_k: int, where: Optional[Dict] = None) -> List[Dict]:
        if self.count() == 0:
            return []
        vector = self.embedder.embed_one(query)
        kwargs = {"where": where} if where else {}
        raw = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
            **kwargs,
        )
        results: List[Dict] = []
        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]
        for i, chunk_id in enumerate(ids):
            distance = float(distances[i]) if i < len(distances) else 1.0
            results.append(
                {
                    "chunk_id": chunk_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distance,
                    # cosine distance → 相似度，用于置信度
                    "dense_sim": max(0.0, min(1.0, 1.0 - distance)),
                }
            )
        return results

    def delete_by_doc(self, doc_id: str) -> None:
        self.collection.delete(where={"doc_id": doc_id})

    def chunks_of(self, doc_id: str) -> int:
        try:
            return len(self.collection.get(where={"doc_id": doc_id}, include=[])["ids"])
        except Exception:
            return 0

    def page_text(self, doc_id: str, page: int) -> str:
        """按页取回完整原文（同页 chunk 按 chunk_idx 拼接）。"""
        try:
            raw = self.collection.get(
                where={"$and": [{"doc_id": doc_id}, {"page": int(page)}]},
                include=["documents", "metadatas"],
            )
        except Exception:
            return ""
        rows = list(zip(raw.get("documents") or [], raw.get("metadatas") or []))
        rows.sort(key=lambda row: int((row[1] or {}).get("chunk_idx", 0)))
        return "\n".join(document for document, _ in rows if document)

    def iter_all(self) -> Iterable[Dict]:
        total = self.count()
        offset = 0
        while offset < total:
            raw = self.collection.get(
                include=["documents", "metadatas"], limit=1000, offset=offset
            )
            ids = raw.get("ids") or []
            if not ids:
                break
            documents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []
            for i, chunk_id in enumerate(ids):
                yield {
                    "chunk_id": chunk_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                }
            offset += len(ids)

    def chunk_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self.iter_all():
            doc_id = (row.get("metadata") or {}).get("doc_id")
            if doc_id:
                counts[doc_id] = counts.get(doc_id, 0) + 1
        return counts

    @property
    def description(self) -> str:
        return self.embedder.description


# ─────────────────────────────────────────────────────────
# 文档台账
# ─────────────────────────────────────────────────────────
class DocStore:
    """documents.json：doc_id → {source, file_hash, path, chunk_count, indexed_at}"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._docs: Dict[str, Dict] = self._load()

    def _load(self) -> Dict[str, Dict]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except Exception:
            logger.warning("文档台账损坏，已忽略：%s", self.path)
            return {}
        docs = data.get("documents") if isinstance(data, dict) else None
        return docs if isinstance(docs, dict) else {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"version": 1, "documents": self._docs}, ensure_ascii=False, indent=2),
            "utf-8",
        )
        os.replace(tmp, self.path)

    def all(self) -> Dict[str, Dict]:
        return dict(self._docs)

    def get(self, doc_id: str) -> Optional[Dict]:
        return self._docs.get(doc_id)

    def find_by_source(self, source: str) -> Optional[Dict]:
        for doc_id, record in self._docs.items():
            if record.get("source") == source:
                return {**record, "doc_id": doc_id}
        return None

    def upsert(self, doc_id: str, record: Dict) -> None:
        self._docs[doc_id] = record

    def remove(self, doc_id: str) -> None:
        self._docs.pop(doc_id, None)

    def reset(self) -> None:
        self._docs.clear()


# ─────────────────────────────────────────────────────────
# 统一入口
# ─────────────────────────────────────────────────────────
class MixedIndex:
    def __init__(self) -> None:
        self.data_dir = Path(config.index.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dense = DenseIndex()
        self.fts = FTSIndex(Path(config.index.persist_dir) / "fts.db")
        self.vocab = VocabStore(Path(config.index.persist_dir) / "vocab.db")
        self.store = DocStore(Path(config.index.persist_dir) / "documents.json")
        self.parser = DocumentParser()
        self.splitter = DocumentSplitter()
        self._lock = threading.RLock()
        self._warmed = False
        self.migrated = False

    # ── 启动 ──────────────────────────────────────────────
    def warm_up(self) -> Dict:
        with self._lock:
            if config.index.auto_migrate and self._needs_migration():
                self.migrated = self._migrate()
            indexed = self._index_pending_files()
            self._ensure_fts()
            self._ensure_vocab()
            self._warmed = True
            return {
                "documents": len(self.store.all()),
                "chunks": self.dense.count(),
                "fts_ready": self.fts.ready,
                "fts_chunks": self.fts.count(),
                "vocab_entries": self.vocab.count(),
                "migrated": self.migrated,
                "indexed_on_startup": indexed,
            }

    def _ensure_fts(self) -> None:
        """首次升级到 FTS5 时，从向量库补建一次关键词索引。"""
        if self.fts.count() > 0 or self.dense.count() == 0:
            return
        chunks: List[Chunk] = []
        for row in self.dense.iter_all():
            metadata = row.get("metadata") or {}
            chunks.append(
                Chunk(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=str(metadata.get("doc_id", "")),
                    source=str(metadata.get("source", "")),
                    page=int(metadata.get("page", 0) or 0),
                    content=str(row.get("content", "")),
                    metadata=metadata,
                )
            )
        if chunks:
            self.fts.add(chunks)
            logger.info("已从向量库补建关键词索引：%d 个片段", len(chunks))

    def _ensure_vocab(self) -> None:
        """词条表为空时，从已入库文档补抽一次（只解析文本层，不触发 OCR）。"""
        if self.vocab.count() > 0:
            return
        total = 0
        for doc_id, record in self.store.all().items():
            path = Path(record.get("path", ""))
            if not path.is_file():
                continue
            try:
                doc = self.parser.parse(str(path), ocr=False)
            except Exception:
                continue
            entries = extract_entries(doc.pages, doc_id, record.get("source", ""))
            if entries:
                total += self.vocab.replace_doc(doc_id, entries)
        if total:
            logger.info("已补抽词条：%d 条", total)

    def _index_pending_files(self) -> int:
        """把 data/ 里尚未入库的文件补进索引（删库/手工拷贝后也能自愈）。"""
        files = sorted(
            path
            for path in self.data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in config.supported_formats
        )
        if not files:
            return 0

        known_paths = {
            str(Path(record["path"]).resolve())
            for record in self.store.all().values()
            if record.get("path")
        }
        known_hashes = {
            record.get("file_hash") for record in self.store.all().values() if record.get("file_hash")
        }

        indexed = 0
        if files:
            logger.info("检查 data/ 中未入库的文件：%d 个", len(files))
        for path in files:
            try:
                if str(path.resolve()) in known_paths:
                    continue
                file_hash = hashlib.md5(path.read_bytes()).hexdigest()
                if file_hash in known_hashes:
                    continue  # 完全相同的内容已经在库里
                doc = self.parser.parse(str(path))
                self._index_parsed(
                    path=path,
                    doc=doc,
                    source=original_name(path),
                    doc_id=None,
                    old_path=None,
                )
                known_hashes.add(file_hash)
                indexed += 1
            except Exception as exc:
                logger.warning("启动时索引 %s 失败：%s", path.name, exc)
        if indexed:
            logger.info("启动时补建索引：%d 个文件", indexed)
        return indexed

    def _needs_migration(self) -> bool:
        if self.dense.count() == 0:
            return False
        try:
            sample = self.dense.collection.get(limit=1, include=["metadatas"])
        except Exception:
            return False
        metadatas = sample.get("metadatas") or []
        return bool(metadatas) and "doc_id" not in (metadatas[0] or {})

    def _migrate(self) -> bool:
        logger.warning("检测到旧版索引（缺少 doc_id 元数据），开始从 data/ 重建…")
        files = [
            path
            for path in self.data_dir.iterdir()
            if path.is_file() and path.suffix.lower() in config.supported_formats
        ]
        try:
            ids = self.dense.collection.get(include=[])["ids"]
            if ids:
                self.dense.collection.delete(ids=ids)
        except Exception as exc:
            logger.error("清空旧索引失败：%s", exc)
            return False
        self.fts.clear()
        self.vocab.clear()
        self.store.reset()
        self.store.save()

        seen: set[str] = set()
        for path in files:
            try:
                doc = self.parser.parse(str(path))
            except Exception as exc:
                logger.warning("跳过无法解析的文件 %s：%s", path.name, exc)
                continue
            if doc.file_hash in seen:
                continue
            seen.add(doc.file_hash)
            try:
                self._index_parsed(
                    path=path,
                    doc=doc,
                    source=original_name(path),
                    doc_id=None,
                    old_path=None,
                )
            except Exception as exc:
                logger.error("重建 %s 失败：%s", path.name, exc)
        logger.warning("旧索引重建完成：%d 个文档 / %d 个 chunk", len(self.store.all()), self.dense.count())
        return True

    # ── 写入 ──────────────────────────────────────────────
    def add_file(
        self,
        temp_path: str | Path,
        original_name_raw: str,
        progress: Callable[[str, float], None] | None = None,
    ) -> IndexResult:
        """把上传的临时文件纳入索引（同名文档视为更新）"""
        temp_path = Path(temp_path)
        doc = self.parser.parse(str(temp_path))
        source = display_name(original_name_raw)
        if progress:
            progress("解析完成（含 OCR）", 0.2)

        with self._lock:
            existing = self.store.find_by_source(source)
            if (
                existing
                and existing.get("file_hash") == doc.file_hash
                and self.dense.chunks_of(existing["doc_id"]) > 0
            ):
                temp_path.unlink(missing_ok=True)
                if progress:
                    progress("内容未变化，跳过重建", 1.0)
                return IndexResult(
                    doc_id=existing["doc_id"],
                    filename=source,
                    chunk_count=self.dense.chunks_of(existing["doc_id"]),
                    skipped=True,
                )

            result = self._index_parsed(
                path=temp_path,
                doc=doc,
                source=source,
                doc_id=existing["doc_id"] if existing else None,
                old_path=existing.get("path") if existing else None,
                progress=progress,
            )
            return result

    def _index_parsed(
        self,
        path: Path,
        doc: ParsedDocument,
        source: str,
        doc_id: Optional[str],
        old_path: Optional[str],
        progress: Callable[[str, float], None] | None = None,
    ) -> IndexResult:
        doc_id = doc_id or uuid.uuid4().hex[:12]
        if progress:
            progress("切分文档", 0.3)
        chunks = self.splitter.split(doc, doc_id, source)
        if not chunks:
            raise ValueError("文档没有可索引的有效文本片段")

        self.dense.delete_by_doc(doc_id)
        if progress:
            progress(f"向量化 {len(chunks)} 个片段", 0.35)
        self.dense.add(
            chunks,
            on_progress=(
                (lambda done, total: progress("向量化", 0.35 + 0.5 * done / max(total, 1)))
                if progress
                else None
            ),
        )
        self.fts.replace_doc(doc_id, chunks)
        entries = extract_entries(doc.pages, doc_id, source)
        if entries:
            self.vocab.replace_doc(doc_id, entries)
            if progress:
                progress(f"抽取词条 {len(entries)} 条", 0.94)
        if progress:
            progress("写入关键词索引", 0.92)

        dest = self.data_dir / f"{doc_id}_{safe_filename(source)}"
        try:
            if path.resolve() != dest.resolve():
                os.replace(path, dest)
        except OSError as exc:
            logger.warning("归档上传文件失败（索引已建立）：%s", exc)
            dest = path

        if old_path and Path(old_path) != dest and Path(old_path).exists():
            try:
                Path(old_path).unlink()
            except OSError:
                pass

        self.store.upsert(
            doc_id,
            {
                "source": source,
                "file_hash": doc.file_hash,
                "path": str(dest),
                "chunk_count": len(chunks),
                "page_count": doc.page_count,
                "size_bytes": doc.size_bytes,
                "indexed_at": int(time.time()),
            },
        )
        self.store.save()
        if progress:
            progress("完成", 1.0)
        return IndexResult(doc_id=doc_id, filename=source, chunk_count=len(chunks))

    def remove_file(self, doc_id: str) -> Optional[Dict]:
        with self._lock:
            record = self.store.get(doc_id)
            self.dense.delete_by_doc(doc_id)
            if record:
                self.store.remove(doc_id)
                self.store.save()
            self.fts.remove_doc(doc_id)
            self.vocab.remove_doc(doc_id)
            return record

    def rebuild_bm25(self) -> None:
        """兼容旧接口：FTS5 已持久化，无需重建。"""
        return None

    # ── 查询 ──────────────────────────────────────────────
    def list_docs(self) -> List[Dict]:
        counts = self.dense.chunk_counts()
        docs: List[Dict] = []
        for doc_id, record in self.store.all().items():
            chunk_count = counts.get(doc_id, 0)
            docs.append(
                {
                    "doc_id": doc_id,
                    "source": record.get("source", ""),
                    "chunk_count": chunk_count,
                    "page_count": record.get("page_count", 0),
                    "size_bytes": record.get("size_bytes", 0),
                    "indexed_at": record.get("indexed_at", 0),
                    "is_current": chunk_count > 0,
                }
            )
        # 台账里没有、但向量库里存在的孤儿数据（例如手工拷贝过 chroma_db）
        for doc_id, chunk_count in counts.items():
            if doc_id and doc_id not in self.store.all():
                docs.append(
                    {
                        "doc_id": doc_id,
                        "source": f"未登记文档({doc_id[:8]})",
                        "chunk_count": chunk_count,
                        "page_count": 0,
                        "size_bytes": 0,
                        "indexed_at": 0,
                        "is_current": False,
                    }
                )
        docs.sort(key=lambda d: d["indexed_at"], reverse=True)
        return docs

    def doc_version(self) -> str:
        """知识库版本号：索引一变，QA 缓存自动失效。"""
        payload = json.dumps(
            [
                [doc_id, record.get("file_hash", ""), record.get("chunk_count", 0)]
                for doc_id, record in sorted(self.store.all().items())
            ],
            ensure_ascii=False,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()[:12]

    def stats(self) -> Dict:
        return {
            "document_count": len(self.store.all()),
            "chunk_count": self.dense.count(),
            "fts_ready": self.fts.ready,
            "fts_chunks": self.fts.count(),
            "vocab_entries": self.vocab.count(),
            "embedding": self.dense.description,
            "data_dir": str(self.data_dir),
            "index_dir": config.index.persist_dir,
        }

    # ── 文件命名 ──────────────────────────────────────────
    # display_name / safe_filename / original_name 见 app/indexing/naming.py


# 全局单例
mixed_index = MixedIndex()
