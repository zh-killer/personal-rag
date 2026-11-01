"""文档切块：递归语义切分 + 清洗 + 元数据绑定

每个 chunk 都带 doc_id / source / page，检索结果因此可以精确溯源到页码。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ..core.config import config
from .parser import ParsedDocument


@dataclass
class Chunk:
    """切块数据结构"""

    chunk_id: str
    doc_id: str
    source: str
    page: int
    content: str
    metadata: Dict = field(default_factory=dict)


class RecursiveTextSplitter:
    """递归切分器：按分隔符优先级切分，尽量保持语义边界"""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: List[str] | None = None,
    ):
        self.chunk_size = chunk_size or config.chunk.chunk_size
        self.chunk_overlap = min(
            chunk_overlap if chunk_overlap is not None else config.chunk.chunk_overlap,
            max(0, self.chunk_size // 2),
        )
        self.separators = separators or list(config.chunk.separators)

    def split(self, text: str) -> List[str]:
        if not text or not text.strip():
            return []
        if len(text) <= self.chunk_size:
            return [text.strip()]
        return [c.strip() for c in self._split_recursive(text, self.separators) if c.strip()]

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._force_split(text)

        separator, remaining = separators[0], separators[1:]
        if separator == "":
            return self._force_split(text)

        parts = text.split(separator)
        chunks: List[str] = []
        current = ""

        for index, part in enumerate(parts):
            # 把分隔符保留在片段末尾，避免切块后句子被截断
            piece = part + (separator if index < len(parts) - 1 else "")
            if not piece.strip():
                continue
            if len(current) + len(piece) <= self.chunk_size:
                current += piece
            else:
                if current.strip():
                    chunks.extend(self._handle_overflow(current, remaining))
                current = piece

        if current.strip():
            chunks.extend(self._handle_overflow(current, remaining))
        return chunks

    def _handle_overflow(self, chunk: str, separators: List[str]) -> List[str]:
        if len(chunk) <= self.chunk_size:
            return [chunk]
        return self._split_recursive(chunk, separators)

    def _force_split(self, text: str) -> List[str]:
        """按字符硬切（带重叠），兜底没有分隔符的超长文本。"""
        step = max(1, self.chunk_size - self.chunk_overlap)
        return [
            text[i : i + self.chunk_size]
            for i in range(0, len(text), step)
            if text[i : i + self.chunk_size].strip()
        ]


class ChunkCleaner:
    """切块清洗：压空白、去分隔线、去纯标点行"""

    _NOISE_LINE = re.compile(r"^[\s\-_=*~#·•]+$")
    _MULTI_NEWLINE = re.compile(r"\n{3,}")
    _MULTI_SPACE = re.compile(r"[ \t]{2,}")

    @classmethod
    def clean(cls, text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = cls._MULTI_NEWLINE.sub("\n\n", text)
        text = cls._MULTI_SPACE.sub(" ", text)
        lines = [line.strip() for line in text.split("\n")]
        lines = [line for line in lines if line and not cls._NOISE_LINE.match(line)]
        return "\n".join(lines).strip()


class DocumentSplitter:
    """文档 → Chunk 列表：切分 + 清洗 + 合并碎块 + 元数据"""

    def __init__(self) -> None:
        self.splitter = RecursiveTextSplitter()
        self.cleaner = ChunkCleaner()
        self.min_chars = config.chunk.min_chunk_chars

    def split(self, doc: ParsedDocument, doc_id: str, source: str) -> List[Chunk]:
        chunks: List[Chunk] = []
        for page_no, page_text in doc.pages:
            raw_pieces = self.splitter.split(page_text)
            cleaned = [self.cleaner.clean(piece) for piece in raw_pieces]
            for content in self._merge_small([c for c in cleaned if c]):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-{len(chunks)}",
                        doc_id=doc_id,
                        source=source,
                        page=page_no,
                        content=content,
                        metadata={
                            "doc_id": doc_id,
                            "source": source,
                            "page": page_no,
                            "chunk_idx": len(chunks),
                            "chars": len(content),
                        },
                    )
                )

        for chunk in chunks:
            chunk.metadata["total_chunks"] = len(chunks)
        return chunks

    def _merge_small(self, pieces: List[str]) -> List[str]:
        """把过短的块并到相邻块，避免产生没有信息量的小碎片。"""
        merged: List[str] = []
        for piece in pieces:
            if (
                merged
                and len(piece) < self.min_chars
                and len(merged[-1]) + len(piece) + 1 <= int(self.splitter.chunk_size * 1.8)
            ):
                merged[-1] = f"{merged[-1]}\n{piece}"
            else:
                merged.append(piece)

        if len(merged) > 1:
            merged = [piece for piece in merged if len(piece) >= self.min_chars]
        return merged
