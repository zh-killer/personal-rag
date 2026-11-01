"""引用核对：把答案里的 [n] 和对应来源做文本比对

生成模型可能写出「看起来像引用」的编号，但原文并不支持那句话。
这里对每个被引用的来源，取出带该编号的句子，与来源片段做关键词重叠，
低于阈值就标成「未在原文找到」——前端据此提示用户点开核对。
"""
from __future__ import annotations

import re
from typing import Dict, List

from ..indexing.tokenizer import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
_CITATION = re.compile(r"\[(?:来源)?(\d+)\]")
_THRESHOLD = 0.45


def verify_citations(answer: str, chunks: List[Dict]) -> List[Dict]:
    sentences = [item.strip() for item in _SENTENCE_SPLIT.split(answer or "") if item.strip()]
    checks: List[Dict] = []

    for index, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content", ""))
        chunk_tokens = set(tokenize(content))
        cited = [sentence for sentence in sentences if re.search(rf"\[(?:来源)?{index}\]", sentence)]
        if not cited:
            checks.append(
                {"index": index, "cited": False, "verified": False, "overlap": 0.0}
            )
            continue

        best = 0.0
        for sentence in cited:
            cleaned = _CITATION.sub("", sentence)
            tokens = set(tokenize(cleaned))
            if not tokens:
                continue
            best = max(best, len(tokens & chunk_tokens) / len(tokens))

        checks.append(
            {
                "index": index,
                "cited": True,
                "verified": best >= _THRESHOLD,
                "overlap": round(best, 2),
            }
        )

    return checks


def summarize(checks: List[Dict]) -> Dict:
    cited = [item for item in checks if item["cited"]]
    unverified = [item["index"] for item in cited if not item["verified"]]
    return {"cited": len(cited), "unverified": unverified}
