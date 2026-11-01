"""离线 OCR：图片 / 扫描件 → 文本

优先 rapidocr-onnxruntime（中英文模型随包，纯离线），
其次 pytesseract（需要系统里装了 tesseract 可执行文件）。
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_engine = None
_backend: str | None = None


def _load_backend() -> str:
    global _engine, _backend
    if _backend is not None:
        return _backend

    try:
        from rapidocr_onnxruntime import RapidOCR

        _engine = RapidOCR()
        _backend = "rapidocr"
        logger.info("OCR 引擎：rapidocr-onnxruntime")
        return _backend
    except Exception as exc:  # 未安装 / 模型加载失败
        logger.info("rapidocr 不可用：%s", exc)

    try:
        import pytesseract  # noqa: F401

        if shutil.which("tesseract"):
            _backend = "tesseract"
            logger.info("OCR 引擎：tesseract")
        else:
            _backend = ""
            logger.warning("pytesseract 已安装但找不到 tesseract 可执行文件")
    except ImportError:
        _backend = ""
    return _backend


def available() -> bool:
    return bool(_load_backend())


def image_to_text(image_path: str | Path) -> str:
    backend = _load_backend()
    if backend == "rapidocr":
        result, _ = _engine(str(image_path))
        lines = [item[1] for item in (result or []) if item and len(item) > 1]
        return "\n".join(line for line in lines if line).strip()

    if backend == "tesseract":
        import pytesseract
        from PIL import Image

        return pytesseract.image_to_string(Image.open(image_path), lang="chi_sim+eng").strip()

    raise RuntimeError(
        "未安装 OCR 引擎，无法处理扫描件/图片：请执行 pip install rapidocr-onnxruntime"
    )
