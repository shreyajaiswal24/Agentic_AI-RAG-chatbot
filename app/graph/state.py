
from dataclasses import dataclass
from typing import Literal, TypedDict

from app.retrieval.models import RetrievedChunk

Route = Literal["generate", "refuse"]
ConfidenceLevel = Literal["high", "medium", "low", "none"]


@dataclass(frozen=True)
class Confidence:
    score: float              # supported_claims / total_claims, in [0, 1]
    level: ConfidenceLevel    # bucketed score for quick reading
    verdict: str              # fully_supported | partially_supported | not_supported | no_answer
    supported_claims: int
    total_claims: int
    reason: str


class RAGState(TypedDict, total=False):
    question: str
    retrieved_chunks: list[RetrievedChunk]
    relevance_score: float   # best cosine similarity among retrieved chunks (retrieval quality)
    is_relevant: bool        # relevance_score >= threshold
    route: Route             # which branch check_relevance picked
    answer: str
    grounded: bool           # True only when the LLM answered from the context
    confidence: Confidence   # evidence-support score from the grade node (answer quality)
