"""Unit tests for the gateway text chunker."""

from services.gateway.src.ingestion.chunker import chunk_text


class TestChunkText:
    def test_basic_chunking(self):
        text = " ".join(["word"] * 100)
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        assert len(chunks) > 1
        assert all(c["chunk_index"] == i for i, c in enumerate(chunks))

    def test_empty_text_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text_single_chunk(self):
        text = "Hello world this is short"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0]["text"] == text

    def test_preserves_section_headings(self):
        text = "# Introduction\nSome intro text.\n# Methods\nSome methods text."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        sections = {c["section"] for c in chunks}
        assert "Introduction" in sections
        assert "Methods" in sections

    def test_overlap_produces_more_chunks(self):
        text = " ".join(["word"] * 100)
        no_overlap = chunk_text(text, chunk_size=30, chunk_overlap=0)
        with_overlap = chunk_text(text, chunk_size=30, chunk_overlap=10)
        assert len(with_overlap) > len(no_overlap)

    def test_chunk_index_sequential(self):
        text = " ".join(["word"] * 200)
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=5)
        for i, c in enumerate(chunks):
            assert c["chunk_index"] == i
