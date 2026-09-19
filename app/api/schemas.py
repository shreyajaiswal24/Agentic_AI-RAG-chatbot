"""Pydantic models defining the HTTP contract."""

from pydantic import BaseModel, Field

from app.graph.state import ConfidenceLevel, Route

SCORE_DESCRIPTION = (
    "Two scores are returned. relevance_score = cosine similarity between the question and the best "
    "retrieved chunk (retrieval quality; answers are only generated when it meets RELEVANCE_THRESHOLD). "
    "confidence.score = fraction of the answer's factual claims that a second, independent LLM pass found "
    "explicitly supported by the retrieved excerpts (answer quality, 0-1). Neither is a probability that "
    "the answer is correct in the real world; both are about the eBook text."
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=["What is Agentic AI?"])


class ChunkOut(BaseModel):
    chunk_id: str
    page: int
    source: str
    score: float = Field(description="Cosine similarity of this chunk to the question")
    text: str


class ConfidenceOut(BaseModel):
    score: float = Field(description="supported_claims / total_claims, judged against the retrieved excerpts")
    level: ConfidenceLevel = Field(description="high >= 0.9, medium >= 0.6, low < 0.6, none = no answer generated")
    verdict: str = Field(description="fully_supported | partially_supported | not_supported | no_answer")
    supported_claims: int
    total_claims: int
    reason: str = Field(description="Judge's explanation, naming any unsupported claim")


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(description="True only when the answer was produced from retrieved context")
    route: Route = Field(description="Graph branch taken: 'generate' (LLM answered) or 'refuse' (relevance gate failed)")
    relevance_score: float = Field(description="Retrieval similarity of the best chunk (cosine)")
    confidence: ConfidenceOut
    score_description: str = SCORE_DESCRIPTION
    retrieved_chunks: list[ChunkOut]


class HealthResponse(BaseModel):
    status: str
    index: str
    namespace: str
    llm_provider: str
    llm_model: str
