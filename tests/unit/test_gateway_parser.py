"""Unit tests for the gateway document parser."""

import pytest

from services.gateway.src.ingestion.parser import (
    parse_file,
    parse_html,
    parse_markdown,
    parse_plain_text,
)


class TestParseMarkdown:
    def test_extracts_title_from_h1(self):
        content = b"# My Document\n\nSome content here."
        title, text = parse_markdown(content)
        assert title == "My Document"
        assert "Some content here." in text

    def test_fallback_title_when_no_heading(self):
        content = b"Just some text without headings."
        title, text = parse_markdown(content)
        assert title == "Untitled"

    def test_empty_content(self):
        title, text = parse_markdown(b"")
        assert title == "Untitled"


class TestParsePlainText:
    def test_uses_first_line_as_title(self):
        content = b"Title Line\nSecond line\nThird line"
        title, text = parse_plain_text(content)
        assert title == "Title Line"
        assert "Third line" in text

    def test_truncates_long_title(self):
        long_line = b"x" * 200 + b"\nrest"
        title, _ = parse_plain_text(long_line)
        assert len(title) <= 100


class TestParseHtml:
    def test_extracts_title_tag(self):
        content = b"<html><head><title>My Page</title></head><body><p>Hello</p></body></html>"
        title, text = parse_html(content)
        assert title == "My Page"
        assert "Hello" in text

    def test_strips_script_and_style(self):
        content = b"<html><body><script>alert(1)</script><style>.x{}</style><p>Content</p></body></html>"
        title, text = parse_html(content)
        assert "alert" not in text
        assert "Content" in text

    def test_no_title_tag(self):
        content = b"<html><body><p>No title</p></body></html>"
        title, _ = parse_html(content)
        assert title == "Untitled"


class TestParseFile:
    def test_dispatches_markdown(self):
        title, text = parse_file("readme.md", b"# Hello\nWorld")
        assert title == "Hello"

    def test_dispatches_txt(self):
        title, text = parse_file("notes.txt", b"My notes\nLine 2")
        assert title == "My notes"

    def test_dispatches_html(self):
        title, text = parse_file("page.html", b"<html><body>Hi</body></html>")
        assert "Hi" in text

    def test_unknown_extension_falls_back(self):
        title, text = parse_file("data.xyz", b"some data")
        assert "some data" in text
