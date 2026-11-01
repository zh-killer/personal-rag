"""文档解析：PDF / Markdown / 纯文本 → 分页纯文本

设计要点：
- 保留页码（切块时写入元数据，引用可以定位到页）；
- 文本编码自动回退（UTF-8 / UTF-8-BOM / GBK / GB18030），避免中文 Windows 上的 txt 乱码；
- 图片与扫描件不做「假装支持」：明确抛出可读的错误，由 API 转成 400。
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import fitz  # PyMuPDF

from ..core.config import config

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """解析结果：按页组织的纯文本"""

    filename: str
    file_hash: str
    pages: List[Tuple[int, str]] = field(default_factory=list)
    format: str = ""
    size_bytes: int = 0
    metadata: Dict = field(default_factory=dict)

    @property
    def content(self) -> str:
        return "\n\n".join(text for _, text in self.pages).strip()

    @property
    def page_count(self) -> int:
        return len(self.pages)


class PDFParser:
    """PDF 解析器（PyMuPDF）"""

    @staticmethod
    def extract_pages(file_path: str, ocr: bool = False) -> List[Tuple[int, str]]:
        try:
            doc = fitz.open(file_path)
        except Exception as exc:  # 损坏 / 加密 / 非 PDF
            raise ValueError(f"无法打开 PDF：{exc}") from exc

        if doc.needs_pass:
            doc.close()
            raise ValueError("PDF 已加密，请先解除密码保护再上传")

        pages: List[Tuple[int, str]] = []
        try:
            for index, page in enumerate(doc, start=1):
                try:
                    text = page.get_text("text")
                except Exception:
                    text = ""
                # 没有文字层的页（扫描件）→ OCR
                if not text.strip() and ocr:
                    text = PDFParser._ocr_page(page)
                if text.strip():
                    pages.append((index, text))
        finally:
            doc.close()
        return pages

    @staticmethod
    def _ocr_page(page) -> str:
        from .ocr import available, image_to_text

        if not available():
            return ""
        temp_path = ""
        try:
            pix = page.get_pixmap(dpi=200)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
                temp_path = handle.name
            pix.save(temp_path)
            return image_to_text(temp_path)
        except Exception as exc:
            logger.warning("第 %s 页 OCR 失败：%s", page.number + 1, exc)
            return ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


class MarkdownParser:
    """Markdown 解析器：去掉图片/链接语法，保留正文"""

    _FRONT_MATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.DOTALL)

    @classmethod
    def extract_text(cls, text: str) -> str:
        text = cls._FRONT_MATTER.sub("", text)
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"[图片: \1]", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
        text = re.sub(r"```[^\n]*\n", "", text)
        text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
        return text.strip()


class DocumentParser:
    """统一文档解析入口"""

    def parse(self, file_path: str, ocr: bool | None = None) -> ParsedDocument:
        path = Path(file_path)
        if not path.is_file():
            raise ValueError(f"文件不存在：{file_path}")

        suffix = path.suffix.lower()
        raw = path.read_bytes()
        file_hash = hashlib.md5(raw).hexdigest()
        ocr_enabled = config.ocr_enabled if ocr is None else ocr

        if suffix == ".pdf":
            pages = PDFParser.extract_pages(str(path), ocr=ocr_enabled)
        elif suffix in (".md", ".markdown"):
            pages = [(1, MarkdownParser.extract_text(self._read_text(raw)))]
        elif suffix == ".txt":
            pages = [(1, self._read_text(raw).strip())]
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            from .ocr import available, image_to_text

            if not available():
                raise ValueError(
                    "图片需要 OCR 才能入库：请执行 pip install rapidocr-onnxruntime 后重启服务"
                )
            pages = [(1, image_to_text(str(path)))]
        else:
            raise ValueError(f"不支持的文件格式：{suffix}")

        pages = [(no, text) for no, text in pages if text.strip()]
        if not pages:
            if suffix == ".pdf":
                raise ValueError(
                    "该 PDF 没有可提取的文本（扫描件），且 OCR 没识别出内容。"
                    "请确认已安装 rapidocr-onnxruntime，或换更清晰的扫描件。"
                )
            raise ValueError("文档内容为空")

        return ParsedDocument(
            filename=path.name,
            file_hash=file_hash,
            pages=pages,
            format=suffix,
            size_bytes=len(raw),
            metadata={"format": suffix, "size_bytes": len(raw), "page_count": len(pages)},
        )

    @staticmethod
    def _read_text(raw: bytes) -> str:
        """按常见中文编码依次尝试解码。"""
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "big5"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="replace")
