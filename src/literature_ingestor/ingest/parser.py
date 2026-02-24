"""PDF → structured Markdown using Marker."""
import re
from dataclasses import dataclass, field
from pathlib import Path

from literature_ingestor.config import settings

# Load Marker models once for the lifetime of the process
_model_dict = None


def _get_models():
    global _model_dict
    if _model_dict is None:
        from marker.models import create_model_dict
        _model_dict = create_model_dict(device=settings.marker_device)
    return _model_dict


@dataclass
class ParsedPaper:
    markdown: str
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None
    abstract: str | None = None


def parse_pdf(pdf_path: Path) -> ParsedPaper:
    """Convert PDF to Markdown and extract coarse metadata."""
    from marker.converters.pdf import PdfConverter
    from marker.output import text_from_rendered

    converter = PdfConverter(artifact_dict=_get_models())
    rendered = converter(str(pdf_path))
    markdown, _, _ = text_from_rendered(rendered)

    return ParsedPaper(
        markdown=markdown,
        title=_extract_title(markdown),
        authors=_extract_authors(markdown),
        year=_extract_year(markdown),
        doi=_extract_doi(markdown),
        abstract=_extract_abstract(markdown),
    )


# ── Heuristic metadata extraction ────────────────────────────────────────────

def _extract_title(md: str) -> str | None:
    """First H1 heading is usually the title."""
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _extract_authors(md: str) -> str | None:
    """Look for an 'Authors' section or a line following the title."""
    match = re.search(r"(?i)^#+\s*authors?\s*\n(.+)", md, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _extract_year(md: str) -> int | None:
    years = re.findall(r"\b(19|20)\d{2}\b", md[:2000])
    if years:
        return int(years[0])
    return None


def _extract_doi(md: str) -> str | None:
    match = re.search(r"10\.\d{4,9}/[^\s\"'<>]+", md)
    if match:
        return match.group(0).rstrip(".,;)")
    return None


def _extract_abstract(md: str) -> str | None:
    match = re.search(
        r"(?i)#+\s*abstract\s*\n([\s\S]+?)(?=\n#+\s|\Z)",
        md,
    )
    if match:
        return match.group(1).strip()[:2000]
    return None
