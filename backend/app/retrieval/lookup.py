"""词条直查：问题里出现词表里的单词时，直接返回词条与页码，不调用大模型。

这是结构化抽取带来的「零成本精确查询」：
- 命中就答，延迟几毫秒；
- 答案直接来自入库时抽好的表，不会被生成模型改写或漏掉。
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

from ..indexing.vocab import VocabStore

_WORDS = re.compile(r"[A-Za-z][A-Za-z'\-]{1,}")

_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "was", "were", "be", "what", "which", "who", "how", "why", "does", "do",
    "did", "mean", "meaning", "word", "please", "tell", "me", "about", "this",
    "that", "these", "those", "it", "its", "with", "from", "by", "how",
}

_CUES = (
    "意思", "释义", "含义", "音标", "怎么读", "读音", "翻译", "是什么",
    "第几页", "哪一页", "词性", "中文",
)

# 带这些词的问题多半在问知识/原理，即使含英文术语也不要走词条直查
_KNOWLEDGE_HINTS = (
    "特性", "原理", "区别", "流程", "模块", "为什么", "如何", "怎么",
    "作用", "影响", "对比", "步骤", "实现", "结构", "机制", "优缺点",
)


def try_vocab_answer(
    question: str, store: VocabStore, doc_id: str | None = None
) -> Optional[Dict]:
    if store.count() == 0:
        return None
    if any(hint in question for hint in _KNOWLEDGE_HINTS):
        return None

    words: List[str] = []
    for raw in _WORDS.findall(question):
        word = raw.strip("'-")
        if len(word) < 2 or word.lower() in _STOP:
            continue
        if word.lower() not in {item.lower() for item in words}:
            words.append(word)
    if not words or len(words) > 3:
        return None

    hits: List[Dict] = []
    for word in words:
        hits.extend(store.lookup(word, doc_id=doc_id))
    if not hits:
        return None

    normalized = question.strip().strip("?？。.!！ ")
    is_single_word = len(words) == 1 and normalized.lower() == words[0].lower()
    if not any(cue in question for cue in _CUES) and not is_single_word:
        return None

    unique: List[Dict] = []
    seen = set()
    for hit in hits:
        key = (hit["word"].lower(), hit["page"], hit["meaning"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(hit)

    lines: List[str] = []
    sources: List[Dict] = []
    for index, hit in enumerate(unique[:5], start=1):
        phonetic = f" [{hit['phonetic']}]" if hit["phonetic"] else ""
        pos = f" {hit['pos']}." if hit["pos"] else ""
        lines.append(
            f"[{index}] **{hit['word']}**{phonetic}{pos} {hit['meaning']}"
            f"（{hit['source']} 第 {hit['page']} 页）"
        )
        sources.append(
            {
                "index": index,
                "chunk_id": "",
                "doc_id": hit["doc_id"],
                "source": hit["source"],
                "page": hit["page"],
                "snippet": f"{hit['word']} {hit['meaning']}",
                "terms": [hit["word"]],
            }
        )

    answer = "\n\n".join(lines)
    if len(unique) > 1:
        answer = "找到这些词条：\n\n" + answer
    answer += "\n\n> 直接命中词条表，未调用大模型。"
    return {"answer": answer, "sources": sources, "confidence": 1.0}
