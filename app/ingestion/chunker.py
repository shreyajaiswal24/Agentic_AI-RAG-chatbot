"""Page-aware chunking.

Each page is split independently so every chunk maps to exactly one page
number (needed for citations). The trade-off — a paragraph that crosses a
page boundary is split in two — is acceptable for a 60-page ebook whose
sections are mostly page-contained, and the overlap softens the cut.
"""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ingestion.loader import Page

MIN_CHUNK_CHARS = 40  # drop fragments like a lone heading with no content


@dataclass
class Chunk:
    id: str          # e.g. "p019-c02": page 19, second chunk on that page
    text: str
    page: int
    chunk_index: int  # index within the page
    source: str      # file name, so multi-document indexes stay distinguishable


def chunk_pages(pages: list[Page], chunk_size: int, chunk_overlap: int, source: str) -> list[Chunk]:
    """Split cleaned pages into overlapping chunks tagged with page metadata."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try paragraph breaks first, then lines, then sentences, then words.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        for i, piece in enumerate(splitter.split_text(page.text)):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(id=f"p{page.number:03d}-c{i:02d}", text=piece, page=page.number, chunk_index=i, source=source)
            )
    return chunks
