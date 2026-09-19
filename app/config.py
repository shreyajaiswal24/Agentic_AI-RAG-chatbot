"""Application settings loaded from environment variables / .env.

Everything tunable lives here so the rest of the code never touches os.environ.
pydantic-settings validates types at startup, so a missing or malformed value
fails fast with a readable error instead of surfacing mid-request.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Provider keys (only the ones your chosen providers need) ---------
    pinecone_api_key: str
    groq_api_key: str | None = None
    openai_api_key: str | None = None

    # --- LLM --------------------------------------------------------------
    llm_provider: Literal["groq", "openai"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"  # Groq model id; e.g. gpt-4o-mini for openai
    llm_temperature: float = 0.0

    # --- Embeddings -------------------------------------------------------
    embedding_provider: Literal["pinecone", "openai"] = "pinecone"
    embedding_model: str = "llama-text-embed-v2"  # text-embedding-3-small for openai
    embedding_dimension: int = 1024               # 1536 for text-embedding-3-small

    # --- Pinecone ---------------------------------------------------------
    pinecone_index_name: str = "agentic-ai-ebook"
    pinecone_namespace: str = "ebook-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Retrieval / chunking --------------------------------------------
    top_k: int = Field(default=5, ge=1, le=20)
    # Cosine similarity below which we treat retrieval as "nothing relevant".
    # Calibrated against real scores from the ebook (see README > Design decisions).
    relevance_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Knowledge base ---------------------------------------------------
    pdf_path: Path = Path("data/Ebook-Agentic-AI.pdf")
    pdf_url: str = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
    source_name: str = "Ebook-Agentic-AI.pdf"

    @model_validator(mode="after")
    def _check_provider_keys(self) -> "Settings":
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise ValueError("LLM_PROVIDER=groq requires GROQ_API_KEY")
        if "openai" in (self.llm_provider, self.embedding_provider) and not self.openai_api_key:
            raise ValueError("Using the openai provider requires OPENAI_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (parsed once per process)."""
    return Settings()  # type: ignore[call-arg]  # values come from the environment
