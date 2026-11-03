"""SSE 帧格式与几个纯函数。

SSE 一旦格式错（比如正文里的换行直接拼进去），前端会静默丢事件，
所以这里把「换行也被 JSON 转义」这条契约固化下来。
"""
from __future__ import annotations

import json

from app.api.common import history_payload, search_snippet, sse
from app.schemas import ChatTurn


def test_sse_frame_shape():
    frame = sse({"type": "token", "text": "你好"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[len("data: "):].strip()) == {"type": "token", "text": "你好"}


def test_sse_escapes_newlines():
    frame = sse({"type": "token", "text": "第一行\n第二行"})
    # 一个事件必须只占一行 data:
    assert frame.count("\n\n") == 1
    assert json.loads(frame[len("data: "):].strip())["text"] == "第一行\n第二行"


def test_history_payload_keeps_recent_turns_only():
    turns = [ChatTurn(question=f"q{i}", answer=f"a{i}") for i in range(20)]
    payload = history_payload(turns)
    assert len(payload) == 5
    assert payload[-1] == {"question": "q19", "answer": "a19"}


def test_search_snippet_centers_on_term():
    content = "前文" * 100 + "目标关键词" + "后文" * 100
    snippet = search_snippet(content, "目标关键词", width=20)
    assert "目标关键词" in snippet
    assert snippet.startswith("…")
    assert snippet.endswith("…")


def test_search_snippet_falls_back_to_head():
    snippet = search_snippet("一段不含查询词的正文内容", "完全不存在的词", width=10)
    assert snippet == "一段不含查询词的正文内容"[:20]
