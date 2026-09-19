"""Wires the nodes into a LangGraph StateGraph.

    START -> retrieve -> check_relevance --(relevant)--> generate --> grade -> END
                                        \\-(irrelevant)-> refuse  --/

``grade`` fact-checks the answer against the excerpts (the confidence score);
on the refuse path it short-circuits to confidence 0 without an LLM call.

The conditional edge is the reason LangGraph earns its place here: the
relevance decision changes *which code runs*, rather than just setting a flag
that a linear chain would have to inspect later.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import RAGNodes
from app.graph.state import RAGState


def build_graph(nodes: RAGNodes) -> CompiledStateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("check_relevance", nodes.check_relevance)
    graph.add_node("generate", nodes.generate)
    graph.add_node("refuse", nodes.refuse)
    graph.add_node("grade", nodes.grade)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "check_relevance")
    graph.add_conditional_edges(
        "check_relevance",
        nodes.route_after_relevance,
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("generate", "grade")
    graph.add_edge("refuse", "grade")
    graph.add_edge("grade", END)

    return graph.compile()
