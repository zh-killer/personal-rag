"""引用核对：答案里的 [n] 到底有没有原文支撑。

这是「减少看起来有出处」的核心逻辑，值得用单测把行为钉住：
1. 引用了且原文重叠够高 → verified
2. 引用了但原文不支持 → 未核对，进 unverified 列表
3. 压根没引用 → cited=False
"""
from __future__ import annotations

from app.retrieval.verify import summarize, verify_citations

CHUNKS = [
    {"content": "卷积神经网络具有局部连接和权值共享的特性"},
    {"content": "反向传播使用链式法则计算梯度"},
]


def test_verified_when_sentence_matches_source():
    answer = "卷积神经网络具有局部连接和权值共享[1]。"
    checks = verify_citations(answer, CHUNKS)
    assert checks[0]["cited"] is True
    assert checks[0]["verified"] is True
    assert checks[0]["overlap"] >= 0.45


def test_unverified_when_source_does_not_support_sentence():
    answer = "今天天气很好，适合出门散步[1]。"
    checks = verify_citations(answer, CHUNKS)
    assert checks[0]["cited"] is True
    assert checks[0]["verified"] is False
    assert summarize(checks)["unverified"] == [1]


def test_uncited_chunk_is_reported_but_not_unverified():
    checks = verify_citations("只有一句话，没有编号。", CHUNKS)
    assert all(item["cited"] is False for item in checks)
    assert summarize(checks) == {"cited": 0, "unverified": []}


def test_accepts_both_citation_styles():
    answer = "卷积神经网络局部连接[1]。反向传播用链式法则[来源2]。"
    checks = verify_citations(answer, CHUNKS)
    assert [item["cited"] for item in checks] == [True, True]


def test_empty_answer_is_safe():
    checks = verify_citations("", CHUNKS)
    assert len(checks) == len(CHUNKS)
    assert summarize(checks)["cited"] == 0
