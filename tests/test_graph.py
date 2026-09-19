from app.llm.prompts import NOT_FOUND_MESSAGE
from tests.conftest import FakeGrader, make_chunk


def test_relevant_question_goes_through_generate(make_graph):
    graph, retriever, llm, grader = make_graph([make_chunk(0.62), make_chunk(0.41, page=20)])
    result = graph.invoke({"question": "How does Agentic AI perceive its environment?"})

    assert result["route"] == "generate"
    assert result["grounded"] is True
    assert result["relevance_score"] == 0.62
    assert "(Page 19)" in result["answer"]
    assert len(llm.calls) == 1
    # the context handed to the LLM must carry the page label for citations
    human_msg = llm.calls[0][1][1]
    assert "[Chunk p019-c00 | Page 19]" in human_msg
    assert result["confidence"].score == 1.0 and result["confidence"].level == "high"
    assert grader.calls == [(result["question"], result["answer"])]


def test_low_similarity_routes_to_refuse_without_calling_llm(make_graph):
    graph, retriever, llm, grader = make_graph([make_chunk(0.12), make_chunk(0.09)])
    result = graph.invoke({"question": "What is the capital of France?"})

    assert result["route"] == "refuse"
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert result["grounded"] is False
    assert llm.calls == []  # the LLM never ran
    assert grader.calls == []  # nor the judge
    assert result["confidence"].level == "none" and result["confidence"].score == 0.0


def test_empty_retrieval_routes_to_refuse(make_graph):
    graph, _, llm, grader = make_graph([])
    result = graph.invoke({"question": "anything"})
    assert result["route"] == "refuse"
    assert result["relevance_score"] == 0.0
    assert llm.calls == []


def test_llm_not_found_reply_is_reported_as_ungrounded(make_graph):
    # Similarity passes the gate, but the model decides the context is insufficient.
    graph, _, _, grader = make_graph([make_chunk(0.45)], reply=NOT_FOUND_MESSAGE)
    result = graph.invoke({"question": "Who is the CEO of Konverge AI?"})
    assert result["route"] == "generate"
    assert result["grounded"] is False
    assert result["answer"] == NOT_FOUND_MESSAGE
    assert grader.calls == []  # nothing to grade


def test_partially_supported_answer_gets_medium_confidence(make_graph):
    graph, _, _, _ = make_graph([make_chunk(0.5)], grader=FakeGrader(supported=2, total=3))
    c = graph.invoke({"question": "q"})["confidence"]
    assert c.score == 0.667 and c.level == "medium" and c.verdict == "partially_supported"


def test_unsupported_answer_gets_low_confidence(make_graph):
    graph, _, _, _ = make_graph([make_chunk(0.5)], grader=FakeGrader(supported=0, total=2))
    c = graph.invoke({"question": "q"})["confidence"]
    assert c.score == 0.0 and c.level == "low"


def test_threshold_is_inclusive(make_graph):
    graph, _, _, grader = make_graph([make_chunk(0.30)], threshold=0.30)
    assert graph.invoke({"question": "q"})["route"] == "generate"
