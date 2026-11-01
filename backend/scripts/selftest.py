"""不依赖服务、不消耗 API Key 的自检。

用法：python backend/scripts/selftest.py
检查项：中文分词、递归切块、页码元数据、短文档兜底、文件名清洗。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.indexing.parser import ParsedDocument
from app.indexing.splitter import DocumentSplitter, RecursiveTextSplitter
from app.indexing.tokenizer import tokenize


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"  [ok] {message}")


def test_tokenizer() -> None:
    tokens = tokenize("默认切块大小是 500 个字符，BM25 与 RAG")
    check("切块" in tokens, "中文按二元组切分（切块）")
    check("500" in tokens, "数字按整词保留")
    check("rag" in tokens and "bm25" in tokens, "英文转小写并保留整词")
    check("的" not in tokens, "停用词被过滤")


def test_splitter() -> None:
    splitter = RecursiveTextSplitter(chunk_size=120, chunk_overlap=20)
    chunks = splitter.split("第一段讲向量检索。" * 40)
    check(len(chunks) > 1, "长文本被切成多块")
    check(all(len(chunk) <= 240 for chunk in chunks), "没有异常超长块")
    check(all(chunk.strip() for chunk in chunks), "没有空白块")


def test_page_metadata() -> None:
    doc = ParsedDocument(
        filename="a.md",
        file_hash="hash-a",
        pages=[(1, "第一页内容。" * 30), (2, "第二页内容。" * 30)],
        format=".md",
    )
    chunks = DocumentSplitter().split(doc, "doc123", "a.md")
    check({chunk.page for chunk in chunks} == {1, 2}, "页码写入 chunk 元数据")
    check(all(chunk.metadata["doc_id"] == "doc123" for chunk in chunks), "doc_id 写入元数据")
    check(all(chunk.chunk_id.startswith("doc123-") for chunk in chunks), "chunk_id 带文档前缀")


def test_short_document() -> None:
    doc = ParsedDocument(filename="b.md", file_hash="hash-b", pages=[(1, "很短。")], format=".md")
    chunks = DocumentSplitter().split(doc, "doc456", "b.md")
    check(len(chunks) == 1, "短文档不会因为长度过滤变成 0 块")


def test_filename_sanitize() -> None:
    from app.indexing.naming import display_name, repair_mojibake

    check(display_name("..\\..\\evil.pdf") == "evil.pdf", "去掉路径穿越前缀")
    check(
        repair_mojibake("ËÄÁù¼¶ºËÐÄ1500´Ê.pdf") == "四六级核心1500词.pdf",
        "修复 GBK/latin-1 乱码文件名",
    )


if __name__ == "__main__":
    print("PersonalRAG 自检")
    for test in (test_tokenizer, test_splitter, test_page_metadata, test_short_document, test_filename_sanitize):
        test()
    print("全部通过")
