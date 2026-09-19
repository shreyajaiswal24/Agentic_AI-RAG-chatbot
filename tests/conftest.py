"""Fakes shared by the graph and API tests. No network access anywhere here."""

from dataclasses import dataclass

import pytest

from app.graph.builder import build_graph
from app.graph.nodes import RAGNodes
from app.llm.grader import AnswerGrade
from app.retrieval.models import RetrievedChunk


def make_chunk(score: float, page: int = 19, text: str = "Agentic AI starts by perceiving its environment.") -> RetrievedChunk:
    return RetrievedChunk(id=f"p{page:03d}-c00", text=text, page=page, chunk_index=0, source="Ebook-Agentic-AI.pdf", score=score)


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.calls.append(query)
        return self.chunks[:top_k]


@dataclass
class FakeMessage:
    content: str


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[tuple[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return FakeMessage(self.reply)


class FakeGrader:
    def __init__(self, supported: int = 2, total: int = 2) -> None:
        self.supported, self.total = supported, total
        self.calls: list[tuple[str, str]] = []

    def grade(self, question, answer, chunks) -> AnswerGrade:
        self.calls.append((question, answer))
        verdict = "fully_supported" if self.supported == self.total else ("not_supported" if self.supported == 0 else "partially_supported")
        return AnswerGrade(total_claims=self.total, supported_claims=self.supported, verdict=verdict, reason="fake")


@pytest.fixture
def make_graph():
    def _make(chunks, reply="Agentic AI perceives its environment (Page 19).", threshold=0.30, grader=None):
        retriever, llm = FakeRetriever(chunks), FakeLLM(reply)
        grader = grader or FakeGrader()
        graph = build_graph(RAGNodes(retriever, llm, grader, top_k=5, relevance_threshold=threshold))
        return graph, retriever, llm, grader

    return _make
