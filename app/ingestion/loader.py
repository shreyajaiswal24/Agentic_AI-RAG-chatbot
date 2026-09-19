"""PDF loading: one Page object per PDF page so page numbers survive to the API."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.errors import IngestionError


@dataclass
class Page:
    number: int  # 1-based, matches what a reader sees in a PDF viewer
    text: str


def load_pdf(path: Path) -> list[Page]:
    """Extract raw text from every page. Empty pages (cover art) are kept so numbering stays exact."""
    if not path.exists():
        raise IngestionError(f"PDF not found at {path}. Run `python -m scripts.ingest` to download it.")
    try:
        reader = PdfReader(str(path))
        return [Page(number=i, text=page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    except Exception as exc:  # pypdf raises a variety of error types
        raise IngestionError(f"Failed to read {path}: {exc}") from exc
