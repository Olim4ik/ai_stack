"""File format parsers for document ingestion."""

import io

import structlog

logger = structlog.get_logger()


def parse_markdown(content: bytes) -> tuple[str, str]:
    """Parse a Markdown file. Returns (title, text)."""
    text = content.decode("utf-8")
    title = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            break
    return title or "Untitled", text


def parse_plain_text(content: bytes) -> tuple[str, str]:
    """Parse a plain text file. Returns (title, text)."""
    text = content.decode("utf-8")
    first_line = text.strip().splitlines()[0] if text.strip() else "Untitled"
    return first_line[:100], text


def parse_html(content: bytes) -> tuple[str, str]:
    """Parse an HTML file, extract text. Returns (title, text)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(content, "html.parser")
    title_tag = soup.find("title")
    title = title_tag.get_text().strip() if title_tag else "Untitled"

    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # Collapse whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def parse_pdf(content: bytes) -> tuple[str, str]:
    """Parse a PDF file, extract text. Returns (title, text)."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)

    full_text = "\n\n".join(pages)
    # Use first line as title fallback
    first_line = full_text.strip().splitlines()[0] if full_text.strip() else "Untitled"
    title = reader.metadata.title if reader.metadata and reader.metadata.title else first_line[:100]
    return title, full_text


PARSERS = {
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".txt": parse_plain_text,
    ".html": parse_html,
    ".htm": parse_html,
    ".pdf": parse_pdf,
}


def parse_file(filename: str, content: bytes) -> tuple[str, str]:
    """Parse a file based on its extension. Returns (title, text)."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    parser = PARSERS.get(ext)
    if not parser:
        logger.warning("Unsupported file format, treating as plain text", filename=filename)
        return parse_plain_text(content)
    return parser(content)
