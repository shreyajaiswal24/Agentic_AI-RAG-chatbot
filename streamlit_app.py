"""Streamlit chat UI for the Agentic AI eBook RAG chatbot.

Runs the LangGraph pipeline in-process (no FastAPI server needed), so it can be
deployed on Streamlit Community Cloud directly from the GitHub repo.

    streamlit run streamlit_app.py
"""

import os

import streamlit as st

# Streamlit Cloud stores secrets in st.secrets; expose them as environment
# variables so app.config (pydantic-settings) picks them up unchanged.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:  # no secrets.toml when running locally with a .env — that's fine
    pass

from app.config import get_settings  # noqa: E402
from app.errors import RAGError  # noqa: E402
from app.main import build_dependencies  # noqa: E402

SAMPLE_QUESTIONS = [
    "What is Agentic AI and how does it differ from traditional AI tools?",
    "What are the core pillars of an Agentic AI system?",
    "What challenges do multi-agent systems face and how can they be mitigated?",
    "Which industries does the eBook give Agentic AI use cases for?",
    "What is the capital of France?",
    "Who is the CEO of Konverge AI?",
]

st.set_page_config(page_title="Agentic AI eBook Chat", page_icon="📖", layout="centered")


@st.cache_resource(show_spinner="Connecting to Pinecone and Groq…")
def load_graph():
    """Build the compiled graph once per server process."""
    return build_dependencies(get_settings())


def render_result(state: dict) -> None:
    st.markdown(state["answer"])

    conf = state["confidence"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Grounded", "yes" if state["grounded"] else "no")
    c2.metric("Retrieval similarity", f"{state['relevance_score']:.3f}")
    c3.metric("Confidence", f"{conf.score:.2f} · {conf.level}")

    if conf.total_claims:
        st.caption(f"Judge: {conf.supported_claims}/{conf.total_claims} claims supported — {conf.reason}")
    else:
        st.caption(f"Route: {state['route']} — {conf.reason}")

    with st.expander(f"{len(state['retrieved_chunks'])} retrieved chunks (cosine similarity, not confidence)"):
        for chunk in state["retrieved_chunks"]:
            st.markdown(f"**{chunk.id} · page {chunk.page} · score {chunk.score:.3f}**")
            st.text(chunk.text)


# ---- page ------------------------------------------------------------------

st.title("📖 Agentic AI eBook Chat")
st.caption(
    "Answers come only from Konverge AI's *Agentic AI: An Executive's Guide*. "
    "Off-topic questions are refused instead of guessed."
)

try:
    graph = load_graph()
except RAGError as exc:
    st.error(f"Startup failed: {exc}")
    st.stop()
except Exception as exc:  # missing secrets, bad key, network
    st.error(f"Startup failed: {exc}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []  # list of (question, state) tuples

with st.sidebar:
    st.subheader("Sample questions")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, use_container_width=True):
            st.session_state.pending = q
    st.divider()
    settings = get_settings()
    st.caption(f"LLM: `{settings.llm_provider}/{settings.llm_model}`")
    st.caption(f"Embeddings: `{settings.embedding_model}`")
    st.caption(f"Index: `{settings.pinecone_index_name}` / `{settings.pinecone_namespace}`")
    st.caption(f"Relevance threshold: `{settings.relevance_threshold}` · top-k: `{settings.top_k}`")

# replay the conversation so far
for question, state in st.session_state.history:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        render_result(state)

question = st.chat_input("Ask a question about the eBook…") or st.session_state.pop("pending", None)

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving, answering, fact-checking…"):
                state = graph.invoke({"question": question.strip()})
        except RAGError as exc:
            st.error(str(exc))
        else:
            render_result(state)
            st.session_state.history.append((question, state))
