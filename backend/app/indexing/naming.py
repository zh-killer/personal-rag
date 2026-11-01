"""文件名处理：路径清洗、非法字符替换、历史乱码修复

独立成模块，方便自检脚本在不初始化向量库的情况下测试。
"""
from __future__ import annotations

import re
from pathlib import Path


def repair_mojibake(name: str) -> str:
    """修复「GBK 字节被当成 latin-1」的历史文件名（如 ËÄÁù¼¶…）。"""
    if any("\u4e00" <= char <= "\u9fff" for char in name):
        return name
    try:
        candidate = name.encode("latin-1").decode("gbk")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name
    fixed_cjk = sum(1 for char in candidate if "\u4e00" <= char <= "\u9fff")
    return candidate if fixed_cjk > 0 else name


def display_name(raw: str) -> str:
    """只保留文件名（去掉任何路径），并修复乱码。"""
    base = Path(str(raw).replace("\\", "/")).name.strip() or "未命名文档"
    return repair_mojibake(base)


def safe_filename(name: str) -> str:
    """用于磁盘落盘：去掉非法字符并限制长度，保留中文。"""
    cleaned = re.sub(r"[^\w\u4e00-\u9fff.\- ]+", "_", name).strip(" ._")
    return (cleaned[:80] or "document")


def original_name(path: Path) -> str:
    """从 {doc_id}_{原名} 恢复原始文件名。"""
    return repair_mojibake(re.sub(r"^[0-9a-f]{8,12}_", "", path.name))
