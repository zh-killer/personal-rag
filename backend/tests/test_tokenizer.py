"""分词与关键词抽取。

这两条是 BM25 检索的地基：查询和建库必须走同一套 tokenize，
否则「库里存了词、查询打不上分」这种问题很难排查。
"""
from __future__ import annotations

from app.indexing.tokenizer import extract_terms, tokenize


def test_tokenize_lowercases_english_words():
    assert tokenize("The Mortgage is DUE") == ["mortgage", "due"]


def test_tokenize_keeps_chinese_single_chars_and_bigrams():
    tokens = tokenize("我的书")
    assert "书" in tokens          # 单字
    assert "我的" in tokens        # 二元组
    assert "的书" in tokens


def test_tokenize_drops_stopwords():
    tokens = tokenize("我的书")
    assert "的" not in tokens
    assert "我" not in tokens


def test_tokenize_empty_input():
    assert tokenize("") == []
    assert tokenize(None) == []    # type: ignore[arg-type]


def test_extract_terms_strips_question_phrases():
    # 「这个」「是什么意思」都是疑问句噪声，真正的检索词只有 mortgage
    terms = extract_terms("mortgage 这个单词是什么意思")
    assert "mortgage" in terms
    assert "什么意思" not in terms
    assert "这个" not in terms


def test_extract_terms_keeps_acronyms():
    terms = extract_terms("CNN 的三大特性是什么")
    assert "cnn" in terms
    assert "三大" in terms
    assert "什么" not in terms
