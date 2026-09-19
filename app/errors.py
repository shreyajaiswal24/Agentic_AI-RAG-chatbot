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


def describe_exception(exc: BaseException) -> str:
    """Render an exception with up to three levels of underlying cause.

    SDKs often wrap the real network error in a generic message (e.g. Groq's
    "Connection error."); the chained cause is what actually tells you what broke.
    """
    parts = [f"{type(exc).__name__}: {exc}"]
    cause = exc.__cause__ or exc.__context__
    for _ in range(3):
        if cause is None:
            break
        parts.append(f"{type(cause).__name__}: {cause}")
        cause = cause.__cause__ or cause.__context__
    return " <- ".join(parts)
