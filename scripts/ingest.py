"""Offline ingestion: PDF -> clean text -> chunks -> embeddings -> Pinecone.

Usage (from the project root, with .env configured):
    python -m scripts.ingest            # downloads the PDF if missing, indexes it
    python -m scripts.ingest --reset    # wipe the namespace first (clean re-index)
    python -m scripts.ingest --pdf path/to/other.pdf
"""

import argparse
import logging
import sys
from pathlib import Path

import httpx

from app.config import get_settings
from app.errors import RAGError
from app.ingestion.chunker import chunk_pages
from app.ingestion.cleaner import clean_text
from app.ingestion.loader import load_pdf
from app.retrieval.embeddings import get_embeddings
from app.retrieval.vector_store import PineconeVectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("pypdf").setLevel(logging.ERROR)  # silence font-encoding warnings
logger = logging.getLogger("ingest")


def download_pdf(url: str, dest: Path) -> None:
    logger.info("Downloading %s -> %s", url, dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in response.iter_bytes():
                fh.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, help="Path to the PDF (default: PDF_PATH from .env)")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectors in the namespace first")
    args = parser.parse_args()

    settings = get_settings()
    pdf_path = args.pdf or settings.pdf_path
    if not pdf_path.exists():
        download_pdf(settings.pdf_url, pdf_path)

    # 1-3. load, clean, chunk
    pages = load_pdf(pdf_path)
    for page in pages:
        page.text = clean_text(page.text)
    chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap, source=pdf_path.name)
    logger.info("Loaded %d pages -> %d chunks (size=%d, overlap=%d)", len(pages), len(chunks), settings.chunk_size, settings.chunk_overlap)

    # 4-5. embed + store
    store = PineconeVectorStore(settings, get_embeddings(settings))
    store.ensure_index()
    if args.reset:
        store.clear_namespace()
    upserted = store.upsert_chunks(chunks)
    logger.info("Done: %d chunks upserted into '%s' / namespace '%s'", upserted, settings.pinecone_index_name, settings.pinecone_namespace)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RAGError as exc:
        logger.error("%s", exc)
        sys.exit(1)
