
import logging
from typing import Protocol

from app.errors import GenerationError
from app.graph.state import Confidence, ConfidenceLevel, RAGState, Route
from app.llm.grader import Grader
from app.llm.prompts import NOT_FOUND_MESSAGE, build_messages
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class Retriever(Protocol):
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class ChatModel(Protocol):
    def invoke(self, messages: list[tuple[str, str]]) -> object: ...  

class RAGNodes:
    def __init__(
        self, retriever: Retriever, llm: ChatModel, grader: Grader, top_k: int, relevance_threshold: float
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._grader = grader
        self._top_k = top_k
        self._threshold = relevance_threshold

    # 1. retrieve 
    def retrieve(self, state: RAGState) -> RAGState:
        chunks = self._retriever.search(state["question"], top_k=self._top_k)
        logger.info("retrieve: %d chunks, top score=%.3f", len(chunks), chunks[0].score if chunks else 0.0)
        return {"retrieved_chunks": chunks}

    # 2. check_relevance 
    def check_relevance(self, state: RAGState) -> RAGState:
        chunks = state.get("retrieved_chunks", [])
        best = max((c.score for c in chunks), default=0.0)
        is_relevant = best >= self._threshold
        route: Route = "generate" if is_relevant else "refuse"
        logger.info("check_relevance: best=%.3f threshold=%.2f -> %s", best, self._threshold, route)
        return {"relevance_score": round(best, 4), "is_relevant": is_relevant, "route": route}

    @staticmethod
    def route_after_relevance(state: RAGState) -> Route:
        """Conditional-edge function: decides which node runs next."""
        return state["route"]

    # 3a. generate 
    def generate(self, state: RAGState) -> RAGState:
        messages = build_messages(state["question"], state["retrieved_chunks"])
        try:
            response = self._llm.invoke(messages)
        except Exception as exc:
            raise GenerationError(f"LLM call failed: {exc}") from exc

        answer = str(getattr(response, "content", response)).strip()
        # Second grounding layer: the model was told to use this exact sentence
        # when the context is insufficient. Report that honestly.
        grounded = NOT_FOUND_MESSAGE.lower() not in answer.lower()
        return {"answer": answer, "grounded": grounded}

    # 3b. refuse 
    @staticmethod
    def refuse(state: RAGState) -> RAGState:
        """Deterministic refusal — no LLM call, so no chance to hallucinate."""
        return {"answer": NOT_FOUND_MESSAGE, "grounded": False}

    # 4. grade 
    def grade(self, state: RAGState) -> RAGState:
        """LLM-as-judge: how much of the answer is supported by the retrieved excerpts."""
        if not state.get("grounded"):
            return {"confidence": Confidence(0.0, "none", "no_answer", 0, 0, "No answer was generated from the knowledge base.")}

        g = self._grader.grade(state["question"], state["answer"], state["retrieved_chunks"])
        score = g.supported_claims / g.total_claims if g.total_claims else 0.0
        score = round(min(max(score, 0.0), 1.0), 3)
        logger.info("grade: %d/%d claims supported -> %.2f (%s)", g.supported_claims, g.total_claims, score, g.verdict)
        reason = g.reason.strip() or "All claims in the answer are supported by the retrieved excerpts."
        return {
            "confidence": Confidence(score, _level(score), g.verdict, g.supported_claims, g.total_claims, reason)
        }


def _level(score: float) -> ConfidenceLevel:
    if score >= 0.9:
        return "high"
    if score >= 0.6:
        return "medium"
    return "low"
