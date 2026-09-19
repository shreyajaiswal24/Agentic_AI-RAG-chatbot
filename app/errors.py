"""Small exception hierarchy so the API layer can map failures to HTTP codes."""


class RAGError(Exception):
    """Base class for all application errors."""


class ConfigurationError(RAGError):
    """Missing/invalid configuration (e.g. index not created yet)."""


class IngestionError(RAGError):
    """PDF could not be loaded, cleaned or chunked."""


class RetrievalError(RAGError):
    """Embedding or vector-store call failed at query time."""


class GenerationError(RAGError):
    """The LLM call failed."""
