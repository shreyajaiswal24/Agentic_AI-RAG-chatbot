"""Pinecone wrapper using the v10 SDK (documents API).

Pinecone SDK >= 10 (Sept 2026) stores each record as a *document*: an ``_id``,
one or more schema-declared vector fields, and arbitrary metadata fields.
Indexes created this way must be written/read through ``index.documents``;
the older ``index.upsert(vectors=...)`` / ``index.query(vector=...)`` calls
are refused on them. This module is the only place that knows those details.
"""

import logging
from typing import Iterable

from langchain_core.embeddings import Embeddings
from pinecone import Pinecone, SchemaBuilder
from pinecone.models.documents.score_by import DenseVectorQuery

from app.config import Settings
from app.errors import ConfigurationError, RetrievalError
from app.ingestion.chunker import Chunk
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

VECTOR_FIELD = "embedding"
METADATA_FIELDS = ["text", "page", "chunk_index", "source"]
UPSERT_BATCH_SIZE = 100  # well under Pinecone's 1000-doc / request-size caps


class PineconeVectorStore:
    def __init__(self, settings: Settings, embeddings: Embeddings) -> None:
        self._settings = settings
        self._embeddings = embeddings
        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = None  # opened lazily so construction never hits the network

    # ---- index lifecycle (ingestion time) --------------------------------

    def ensure_index(self) -> None:
        """Create the serverless index if it does not exist. Idempotent."""
        s = self._settings
        if self._pc.indexes.exists(name=s.pinecone_index_name):
            logger.info("Index '%s' already exists", s.pinecone_index_name)
            return
        logger.info("Creating index '%s' (%d dims, cosine)", s.pinecone_index_name, s.embedding_dimension)
        schema = (
            SchemaBuilder()
            .add_dense_vector_field(name=VECTOR_FIELD, dimension=s.embedding_dimension, metric="cosine")
            .build()
        )
        # create() blocks until the index is ready by default.
        self._pc.indexes.create(
            name=s.pinecone_index_name,
            schema=schema,
            deployment={"deployment_type": "managed", "cloud": s.pinecone_cloud, "region": s.pinecone_region},
        )

    def _get_index(self):
        if self._index is None:
            name = self._settings.pinecone_index_name
            if not self._pc.indexes.exists(name=name):
                raise ConfigurationError(
                    f"Pinecone index '{name}' does not exist. Run `python -m scripts.ingest` first."
                )
            self._index = self._pc.index(name=name)
        return self._index

    def clear_namespace(self) -> None:
        """Delete everything in the configured namespace (used by `ingest --reset`)."""
        index = self._get_index()
        ns = self._settings.pinecone_namespace
        try:
            existing = {n.name for page in index.list_namespaces() for n in page.namespaces}
            if ns in existing:
                index.documents.delete(namespace=ns, delete_all=True)
                logger.info("Cleared namespace '%s'", ns)
        except Exception as exc:  # namespace may not exist on a fresh index
            logger.warning("Could not clear namespace '%s': %s", ns, exc)

    def upsert_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert chunks in batches. Returns the number of documents upserted."""
        index = self._get_index()
        total = 0
        for batch in _batched(chunks, UPSERT_BATCH_SIZE):
            vectors = self._embeddings.embed_documents([c.text for c in batch])
            documents = [
                {
                    "_id": c.id,
                    VECTOR_FIELD: vec,
                    "text": c.text,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    "source": c.source,
                }
                for c, vec in zip(batch, vectors)
            ]
            response = index.documents.upsert(namespace=self._settings.pinecone_namespace, documents=documents)
            total += response.upserted_count
            logger.info("Upserted %d/%d chunks", total, len(chunks))
        return total

    def stats(self) -> dict:
        return self._get_index().describe_index_stats().to_dict()

    # ---- query time ------------------------------------------------------

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Embed the query and return the top_k most similar chunks with metadata."""
        try:
            vector = self._embeddings.embed_query(query)
            response = self._get_index().documents.search(
                namespace=self._settings.pinecone_namespace,
                top_k=top_k,
                score_by=[DenseVectorQuery(field=VECTOR_FIELD, values=vector)],
                include_fields=METADATA_FIELDS,
            )
        except ConfigurationError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Vector search failed: {exc}") from exc

        return [
            RetrievedChunk(
                id=doc.id,
                text=str(doc.get("text", "")),
                page=int(doc.get("page", 0)),
                chunk_index=int(doc.get("chunk_index", 0)),
                source=str(doc.get("source", "")),
                score=float(doc.score or 0.0),
            )
            for doc in response.matches
        ]


def _batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]
