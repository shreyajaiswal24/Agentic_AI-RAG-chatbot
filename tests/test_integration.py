"""End-to-end scenarios against the real OpenAI + Pinecone services.

Requires a populated index (`python -m scripts.ingest`) and a .env.
    RUN_INTEGRATION=1 pytest -m integration -v

Each test states the expected behaviour for one of the five required scenarios.
"""

import os

import pytest

from app.config import get_settings
from app.llm.prompts import NOT_FOUND_MESSAGE
from app.main import build_dependencies

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_INTEGRATION") != "1", reason="set RUN_INTEGRATION=1 to run"),
]


@pytest.fixture(scope="module")
def graph():
    return build_dependencies(get_settings())


def ask(graph, question: str) -> dict:
    return graph.invoke({"question": question})


def test_1_clearly_answered(graph):
    """Expected: generate route, grounded, answer cites the page and mentions perception."""
    r = ask(graph, "What are the core pillars of an Agentic AI system, from perception to execution?")
    assert r["route"] == "generate" and r["grounded"]
    assert "perception" in r["answer"].lower()
    assert "page" in r["answer"].lower()
    assert r["confidence"].score >= 0.9 and r["confidence"].level == "high"


def test_2_multi_chunk_synthesis(graph):
    """Expected: answer draws on more than one retrieved page (challenges + mitigations table)."""
    r = ask(graph, "What challenges do multi-agent systems face and how can they be mitigated?")
    assert r["route"] == "generate" and r["grounded"]
    pages = {c.page for c in r["retrieved_chunks"]}
    assert len(pages) >= 2
    assert any(word in r["answer"].lower() for word in ("interoperability", "scalability", "security"))


def test_3_not_in_pdf(graph):
    """Expected: refusal — either via the relevance gate or via the LLM sentinel — and grounded=False."""
    r = ask(graph, "What is the capital of France?")
    assert r["grounded"] is False
    assert r["answer"] == NOT_FOUND_MESSAGE
    assert r["confidence"].level == "none"


def test_4_vague_question(graph):
    """Expected: vague-but-on-topic questions still retrieve relevant material; the answer stays grounded
    (cites pages) rather than turning into generic knowledge about agents."""
    r = ask(graph, "Tell me about agents.")
    assert r["relevance_score"] > 0
    if r["route"] == "generate" and r["grounded"]:
        assert "page" in r["answer"].lower()
    else:
        assert r["answer"] == NOT_FOUND_MESSAGE


def test_5_hallucination_bait(graph):
    """Expected: the question is on-topic (Konverge AI is in the book) so retrieval may pass the gate,
    but the specific facts (CEO name, founding year) are not in the text -> the model must refuse."""
    r = ask(graph, "Who is the CEO of Konverge AI and when was the company founded?")
    assert r["grounded"] is False
    assert r["answer"] == NOT_FOUND_MESSAGE
