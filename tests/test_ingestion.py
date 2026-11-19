"""Unit tests for ingestion module."""

from ingestion import chunk_text, stable_chunk_id


def test_chunk_text_basic():
    text = " ".join(["word"] * 100)
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.split()) <= 50


def test_chunk_text_small_text():
    text = "small text"
    chunks = chunk_text(text, size=50, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == "small text"


def test_chunk_text_empty():
    chunks = chunk_text("", size=50, overlap=10)
    assert chunks == []


def test_chunk_text_whitespace_only():
    chunks = chunk_text("   \n\t  ", size=50, overlap=10)
    assert chunks == []


def test_stable_chunk_id_deterministic():
    id1 = stable_chunk_id("source.md", 0, "content")
    id2 = stable_chunk_id("source.md", 0, "content")
    assert id1 == id2


def test_stable_chunk_id_different_for_different_inputs():
    id1 = stable_chunk_id("source.md", 0, "content")
    id2 = stable_chunk_id("source.md", 1, "content")
    id3 = stable_chunk_id("source.md", 0, "different content")
    assert id1 != id2
    assert id1 != id3
    assert id2 != id3


def test_stable_chunk_id_format():
    chunk_id = stable_chunk_id("source.md", 0, "content")
    assert len(chunk_id) == 32
    assert all(c in "0123456789abcdef" for c in chunk_id)
