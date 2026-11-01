"""把 PDF 的某一页按版面重建成 Markdown（供原文预览用）

直接拼「切块后的纯文本」会丢换行和层次，预览像一堵墙。
这里用 PyMuPDF 的文本块、字号与表格检测重建：
- 字号明显大于正文、且不是公式 → 三级标题
- 以 • · - 开头 → 列表项
- 被拆成多个文本块的行，按「垂直间距 + 上一行结尾没有句号」合并
- 检测到的表格 → Markdown 表格，表格区域内的文本块不再重复输出
"""
from __future__ import annotations

import logging
import re
from statistics import median
from typing import Dict, List, Sequence

import fitz

logger = logging.getLogger(__name__)

_BULLET = re.compile(r"^\s*[•·▪◦●\-]\s*")
_NUMBERED = re.compile(r"^\s*(?:\d{1,2}[.、)]|[（(]\d{1,2}[）)])\s*")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_SENTENCE_END = re.compile(r"[。！？：；!?;:]\s*$")
_FORMULA = re.compile(r"[=<>∑∫√±×÷]|\\frac|\^")
_HEADING_MAX = 34


def _join(previous: str, following: str) -> str:
    if not previous:
        return following
    if not following:
        return previous
    if _CJK.search(previous[-1]) and _CJK.search(following[0]):
        return previous + following
    return f"{previous} {following}"


def _join_lines(lines: Sequence[str]) -> str:
    merged = ""
    for line in lines:
        line = line.strip()
        if line:
            merged = _join(merged, line)
    return merged


def _table_markdown(rows: Sequence[Sequence]) -> str:
    cleaned: List[List[str]] = []
    for row in rows:
        cleaned.append(
            [" ".join(str(cell).split()) if cell is not None else "" for cell in row]
        )
    if len(cleaned) < 2:
        return ""

    width = max(len(row) for row in cleaned)
    if width < 2:
        return ""
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    if not any(cell for row in cleaned for cell in row):
        return ""

    header = cleaned[0]
    if not any(header):
        return ""

    def cell(value: str) -> str:
        return value.replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(value) for value in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in cleaned[1:]:
        lines.append("| " + " | ".join(cell(value) for value in row) + " |")
    return "\n".join(lines)


def _center_inside(bbox: Sequence[float], boxes: Sequence[Sequence[float]]) -> bool:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return any(
        box[0] <= center_x <= box[2] and box[1] <= center_y <= box[3] for box in boxes
    )


def page_markdown(file_path: str, page: int) -> str:
    """返回该页的 Markdown；没有文本层时返回空串。"""
    document = fitz.open(str(file_path))
    try:
        if page < 1 or page > document.page_count:
            raise ValueError("页码超出范围")

        pdf_page = document[page - 1]
        data = pdf_page.get_text("dict")
        raw_blocks = [block for block in data.get("blocks", []) if block.get("type") == 0]

        # 表格先抽出来，表格区域内的文本块后面要跳过，避免重复
        table_items: List[Dict] = []
        table_boxes: List[Sequence[float]] = []
        try:
            found = pdf_page.find_tables()
            for table in found.tables:
                markdown = _table_markdown(table.extract())
                if not markdown:
                    continue
                table_boxes.append(table.bbox)
                table_items.append(
                    {
                        "kind": "table",
                        "text": markdown,
                        "top": float(table.bbox[1]),
                        "bottom": float(table.bbox[3]),
                    }
                )
        except Exception as exc:
            logger.warning("表格检测失败：%s", exc)

        if not raw_blocks and not table_items:
            return ""

        sizes: List[float] = []
        for block in raw_blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if str(span.get("text", "")).strip():
                        sizes.append(float(span.get("size", 0) or 0))
        body_size = median(sizes) if sizes else 10.0

        items: List[Dict] = []
        for block in sorted(raw_blocks, key=lambda item: (round(item["bbox"][1], 1), item["bbox"][0])):
            if _center_inside(block["bbox"], table_boxes):
                continue

            lines: List[str] = []
            max_size = 0.0
            bold = False
            for line in block.get("lines", []):
                text = "".join(span.get("text", "") for span in line.get("spans", []))
                if text.strip():
                    lines.append(text)
                for span in line.get("spans", []):
                    max_size = max(max_size, float(span.get("size", 0) or 0))
                    if "bold" in str(span.get("font", "")).lower():
                        bold = True
            if not lines:
                continue

            text = _join_lines(lines).strip()
            if not text:
                continue

            is_heading = (
                (max_size >= body_size * 1.22 or (bold and len(text) <= 18))
                and len(text) <= _HEADING_MAX
                and not _FORMULA.search(text)
            )
            if is_heading:
                kind = "heading"
            elif _BULLET.match(text):
                kind = "bullet"
                text = _BULLET.sub("", text).strip()
            elif _FORMULA.search(text) and len(text) <= 40:
                kind = "formula"
            elif _NUMBERED.match(text):
                kind = "numbered"
            else:
                kind = "body"

            items.append(
                {
                    "kind": kind,
                    "text": text,
                    "top": float(block["bbox"][1]),
                    "bottom": float(block["bbox"][3]),
                }
            )

        items.extend(table_items)
        items.sort(key=lambda item: (item["top"], item["kind"] == "table"))

        merged: List[Dict] = []
        for item in items:
            if merged and item["top"] - merged[-1]["bottom"] <= 8:
                previous = merged[-1]
                if (
                    item["kind"] == "body"
                    and previous["kind"] in ("body", "numbered")
                    and not _SENTENCE_END.search(previous["text"])
                ):
                    previous["text"] = _join(previous["text"], item["text"])
                    previous["bottom"] = item["bottom"]
                    continue
                if item["kind"] == previous["kind"] == "formula":
                    previous["text"] = _join(previous["text"], item["text"])
                    previous["bottom"] = item["bottom"]
                    continue
            merged.append(dict(item))

        parts: List[str] = []
        for item in merged:
            if item["kind"] == "heading":
                parts.append(f"### {item['text']}")
            elif item["kind"] == "bullet":
                parts.append(f"- {item['text']}")
            elif item["kind"] == "formula":
                parts.append(f"```\n{item['text']}\n```")
            else:
                parts.append(item["text"])

        return "\n\n".join(parts)
    finally:
        document.close()
