"""切块：页码必须跟着块走，否则「页码级溯源」就是假的。"""
from __future__ import annotations

from app.indexing.parser import ParsedDocument
from app.indexing.splitter import ChunkCleaner, DocumentSplitter


def _doc(*pages: tuple[int, str]) -> ParsedDocument:
    return ParsedDocument(filename="a.pdf", file_hash="hash", pages=list(pages))


def test_cleaner_collapses_whitespace_and_drops_rule_lines():
    text = "A  B\r\n\r\n-----\r\nC"
    assert ChunkCleaner.clean(text) == "A B\nC"


def test_cleaner_keeps_meaningful_lines():
    assert ChunkCleaner.clean("第一行\n\n\n第二行") == "第一行\n第二行"


def test_split_keeps_page_number_and_doc_id():
    chunks = DocumentSplitter().split(
        _doc((3, "第一段内容。" * 30), (4, "第二段内容。" * 30)), "doc123", "a.pdf"
    )
    assert [chunk.page for chunk in chunks] == [3, 4]
    assert chunks[0].chunk_id == "doc123-0"
    assert all(chunk.doc_id == "doc123" for chunk in chunks)
    assert all(chunk.metadata["total_chunks"] == len(chunks) for chunk in chunks)


def test_split_long_page_produces_multiple_chunks():
    long_page = "这是一个很长的段落，用来验证切块逻辑。" * 200
    chunks = DocumentSplitter().split(_doc((1, long_page)), "doc1", "a.pdf")
    assert len(chunks) > 1
    assert all(chunk.page == 1 for chunk in chunks)


def test_split_skips_blank_pages():
    chunks = DocumentSplitter().split(_doc((1, "   "), (2, "有内容的一页。" * 20)), "doc1", "a.pdf")
    assert len(chunks) == 1
    assert chunks[0].page == 2
