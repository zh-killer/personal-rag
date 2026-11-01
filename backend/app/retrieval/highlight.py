"""为来源计算「预览时要高亮的词」

直接拿 extract_terms 的二元组会得到「决策」「策树」两段，
在页面上高亮会被切碎。这里把相邻二元组合并回完整词（决策树）。
"""
from __future__ import annotations

import re
from typing import List

from ..indexing.tokenizer import extract_terms

_CJK = re.compile(r"^[\u4e00-\u9fff]+$")


def matched_terms(query: str, content: str, limit: int = 4) -> List[str]:
    if not query:
        return []

    lowered = content.lower()
    raw = [term for term in extract_terms(query) if term and term.lower() in lowered]

    merged: List[str] = []
    for term in raw:
        if (
            merged
            and _CJK.match(term)
            and _CJK.match(merged[-1])
            and len(term) == 2
            and merged[-1][-1] == term[0]
        ):
            merged[-1] = merged[-1] + term[1]
        else:
            merged.append(term)

    unique: List[str] = []
    for term in merged:
        if term not in unique:
            unique.append(term)
    return unique[:limit]
