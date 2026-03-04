"""Text chunking strategies for document ingestion."""

import re

import structlog

logger = structlog.get_logger()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split text into overlapping chunks, preserving section headings.

    Returns a list of dicts with 'text', 'chunk_index', and 'section' keys.
    """
    if not text.strip():
        return []

    sections = _split_into_sections(text)
    chunks = []
    chunk_index = 0

    for section_title, section_text in sections:
        words = section_text.split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text_str = " ".join(chunk_words)

            chunks.append({
                "text": chunk_text_str,
                "chunk_index": chunk_index,
                "section": section_title,
            })
            chunk_index += 1

            # Move forward by (chunk_size - overlap)
            start += chunk_size - chunk_overlap
            if start >= len(words):
                break

    logger.info("Text chunked", total_chunks=len(chunks), chunk_size=chunk_size)
    return chunks


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split text into (section_title, section_text) pairs by markdown headings."""
    heading_pattern = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [("", text)]

    sections = []
    # Text before first heading
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()].strip()))

    for i, match in enumerate(matches):
        title = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((title, section_text))

    return sections
