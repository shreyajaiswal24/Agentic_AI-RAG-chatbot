from fastapi.testclient import TestClient

from app.config import Settings
from app.graph.builder import build_graph
from app.graph.nodes import RAGNodes
from app.llm.prompts import NOT_FOUND_MESSAGE
from app.main import create_app
from tests.conftest import FakeGrader, FakeLLM, FakeRetriever, make_chunk


def _client(chunks, reply="Answer (Page 19)."):
    settings = Settings(pinecone_api_key="test", groq_api_key="test")
    graph = build_graph(RAGNodes(FakeRetriever(chunks), FakeLLM(reply), FakeGrader(), top_k=5, relevance_threshold=0.30))
    return TestClient(create_app(graph=graph, settings=settings))


def test_health():
    with _client([]) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["llm_provider"] == "groq"


def test_ask_returns_full_contract():
    with _client([make_chunk(0.55), make_chunk(0.40, page=20)]) as client:
        resp = client.post("/ask", json={"question": "How does Agentic AI perceive its environment?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Answer (Page 19)."
    assert body["grounded"] is True
    assert body["route"] == "generate"
    assert body["relevance_score"] == 0.55
    assert "cosine similarity" in body["score_description"]
    assert body["confidence"] == {"score": 1.0, "level": "high", "verdict": "fully_supported",
                                  "supported_claims": 2, "total_claims": 2, "reason": "fake"}
    assert [c["page"] for c in body["retrieved_chunks"]] == [19, 20]
    assert body["retrieved_chunks"][0]["chunk_id"] == "p019-c00"


def test_ask_off_topic_refuses():
    with _client([make_chunk(0.11)]) as client:
        body = client.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert body["route"] == "refuse"
    assert body["grounded"] is False
    assert body["answer"] == NOT_FOUND_MESSAGE
    assert body["confidence"]["level"] == "none"


def test_ask_validates_input():
    with _client([]) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        assert client.post("/ask", json={}).status_code == 422


def test_root_serves_chat_page():
    with _client([]) as client:
        resp = client.get("/")
    assert resp.status_code == 200
    assert "Agentic AI eBook Chat" in resp.text
