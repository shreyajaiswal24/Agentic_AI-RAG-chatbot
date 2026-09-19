"""Shared shape for a chunk coming back from the vector store."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    page: int
    chunk_index: int
    source: str
    score: float  # cosine similarity in [-1, 1]; see README for the observed range
