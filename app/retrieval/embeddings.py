"""Embedding model factory.

Two providers behind LangChain's ``Embeddings`` interface so ingestion and
query time share one implementation:

* ``pinecone`` (default) — Pinecone-hosted ``llama-text-embed-v2`` (1024-d).
  No extra vendor: the same key that stores vectors also produces them.
  The model is *asymmetric*, so passages and queries are embedded with
  different ``input_type`` values — mixing them up silently degrades recall.
* ``openai`` — ``text-embedding-3-small`` (1536-d) via langchain-openai.
"""

from langchain_core.embeddings import Embeddings
from pinecone import Pinecone

from app.config import Settings
from app.errors import ConfigurationError

EMBED_BATCH_SIZE = 96  # Pinecone inference accepts up to 96 inputs per call


class PineconeInferenceEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str, dimension: int) -> None:
        self._pc = Pinecone(api_key=api_key)
        self._model = model
        self._dimension = dimension

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            result = self._pc.inference.embed(
                model=self._model,
                inputs=texts[i : i + EMBED_BATCH_SIZE],
                parameters={"input_type": input_type, "truncate": "END", "dimension": self._dimension},
            )
            vectors.extend(list(e.values) for e in result)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, input_type="passage")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], input_type="query")[0]


def get_embeddings(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ConfigurationError("EMBEDDING_PROVIDER=openai requires OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)

    return PineconeInferenceEmbeddings(
        api_key=settings.pinecone_api_key, model=settings.embedding_model, dimension=settings.embedding_dimension
    )
