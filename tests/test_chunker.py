import pytest

from app.ingestion.chunker import chunk_pages
from app.ingestion.loader import Page


def _pages():
    long_text = " ".join(f"Sentence number {i} about agentic systems." for i in range(60))
    return [Page(number=1, text=""), Page(number=2, text="Short heading"), Page(number=3, text=long_text)]


def test_chunks_carry_page_metadata_and_ids():
    chunks = chunk_pages(_pages(), chunk_size=300, chunk_overlap=50, source="book.pdf")
    assert chunks, "expected at least one chunk"
    assert all(c.page == 3 for c in chunks)  # page 1 empty, page 2 below MIN_CHUNK_CHARS
    assert chunks[0].id == "p003-c00"
    assert chunks[1].id == "p003-c01"
    assert all(c.source == "book.pdf" for c in chunks)


def test_chunks_respect_size_limit():
    chunks = chunk_pages(_pages(), chunk_size=300, chunk_overlap=50, source="book.pdf")
    assert all(len(c.text) <= 300 for c in chunks)


def test_overlap_shares_text_between_neighbours():
    chunks = chunk_pages(_pages(), chunk_size=300, chunk_overlap=100, source="book.pdf")
    tail = chunks[0].text[-40:]
    assert tail in chunks[1].text


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_pages(_pages(), chunk_size=100, chunk_overlap=100, source="x")
