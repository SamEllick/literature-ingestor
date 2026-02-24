"""Section-aware Markdown chunker for academic papers."""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    section: str
    chunk_index: int
    paper_id: int | None = None


_HEADING = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)

# Sections to skip entirely (noisy for RAG)
_SKIP_SECTIONS = {
    "references", "bibliography", "acknowledgements", "acknowledgments",
    "appendix", "supplementary", "funding", "conflicts of interest",
    "author contributions",
}


def chunk_markdown(
    markdown: str,
    paper_id: int | None = None,
    max_tokens: int = 512,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split Markdown into section-aware chunks with token-based size limits."""
    sections = _split_by_headings(markdown)
    chunks: list[Chunk] = []
    idx = 0

    for section_name, section_text in sections:
        if section_name.lower().strip() in _SKIP_SECTIONS:
            continue
        for chunk_text in _window(section_text, max_tokens, overlap_tokens):
            chunks.append(Chunk(
                text=chunk_text.strip(),
                section=section_name,
                chunk_index=idx,
                paper_id=paper_id,
            ))
            idx += 1

    return chunks


def _split_by_headings(markdown: str) -> list[tuple[str, str]]:
    """Return list of (heading, content) pairs."""
    parts: list[tuple[str, str]] = []
    current_heading = "preamble"
    current_lines: list[str] = []

    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            if current_lines:
                parts.append((current_heading, "\n".join(current_lines)))
            current_heading = m.group(2).strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        parts.append((current_heading, "\n".join(current_lines)))

    return parts


def _window(text: str, max_tokens: int, overlap: int) -> list[str]:
    """Naive whitespace-token windowing with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks
