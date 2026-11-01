"""中英文混合分词（BM25 稀疏检索专用）

不依赖 jieba：
- 中文按「单字 + 二元组（bigram）」切分，中文没有空格也能正确召回；
- 英文 / 数字按整词切分并小写；
- 过滤极少量高频停用词，降低噪声。

查询和建库必须用同一套 tokenize，否则 BM25 永远打不上分。
"""
from __future__ import annotations

import re

_CJK_RANGE = r"\u4e00-\u9fff"
_TOKEN_RE = re.compile(rf"[{_CJK_RANGE}]+|[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "而", "也", "就", "都",
    "这", "那", "我", "你", "他", "她", "它", "们", "吗", "呢", "吧", "啊",
    "一个", "一种", "什么", "怎么", "如何", "哪些", "请问", "一下", "可以",
    "the", "a", "an", "of", "to", "in", "is", "are", "and", "or", "for",
}


def tokenize(text: str) -> list[str]:
    """把任意文本切成 BM25 可用的 token 列表。"""
    if not text:
        return []
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(text):
        piece = match.group().lower()
        if re.fullmatch(rf"[{_CJK_RANGE}]+", piece):
            # 中文：单字 + 相邻二元组
            tokens.extend(piece)
            tokens.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        else:
            tokens.append(piece)
    return [t for t in tokens if t not in _STOPWORDS]


# 问句里的“虚词”片段，只影响关键词检索，不影响向量检索
_NOISE_PHRASES = [
    "是什么意思", "什么意思", "是什么", "有什么", "有哪些", "讲了什么", "说了什么",
    "说的是什么", "怎么样", "为什么", "怎么用", "第几页", "几页", "请问", "帮我",
    "帮忙", "告诉我", "查一下", "找一下", "搜一下", "搜索", "关于", "有关", "相关",
    "这个", "那个", "这些", "那些", "里面", "中的", "文档", "资料", "知识库",
    "一下", "具体", "内容", "什么", "哪些", "如何", "多少", "怎么", "是否",
]
_NOISE_PHRASES.sort(key=len, reverse=True)


def extract_terms(text: str) -> list[str]:
    """从自然语言问句里抽取“检索关键词”，用于 BM25 与精确命中。

    例如「mortgage 这个单词是什么意思」→ ["mortgage", "单词", "单词"] 级别的噪声被剔除后，
    只剩 mortgage 这类真正的检索词，避免疑问句里的高频字把稀有词淹没。
    中文只保留二元组（单字噪声太大），英文保留长度 >= 2 的整词。
    """
    if not text:
        return []
    cleaned = text
    for phrase in _NOISE_PHRASES:
        cleaned = cleaned.replace(phrase, " ")

    terms: list[str] = []
    for match in _TOKEN_RE.finditer(cleaned):
        piece = match.group().lower()
        if re.fullmatch(rf"[{_CJK_RANGE}]+", piece):
            terms.extend(piece[i : i + 2] for i in range(len(piece) - 1))
        elif len(piece) >= 2 and piece not in _STOPWORDS:
            terms.append(piece)
    return terms
