"""Prompt templates and context formatting for the generation step.

Grounding strategy (three layers, see README > Design decisions):
  1. The relevance gate in the graph refuses before the LLM is even called.
  2. This system prompt forbids outside knowledge and gives the model an
     explicit, easy escape hatch (NOT_FOUND_MESSAGE) so refusing is cheaper
     than inventing.
  3. The node checks the reply for that sentinel so the API can report
     grounded=false honestly.
"""

from app.retrieval.models import RetrievedChunk

NOT_FOUND_MESSAGE = "The information is not available in the provided knowledge base."

SYSTEM_PROMPT = f"""You are a question-answering assistant for a single document: the eBook \
"Agentic AI: An Executive's Guide" by Konverge AI.

Rules — follow them strictly:
1. Answer ONLY using the context excerpts provided below. Do not use prior knowledge, \
even if you are confident about the answer.
2. If the context does not contain enough information to answer the question, reply with \
exactly this sentence and nothing else: "{NOT_FOUND_MESSAGE}"
3. Never guess names, numbers, dates or facts that are not explicitly in the context.
4. Cite the page number(s) you relied on in parentheses, e.g. (Page 19).
5. Be concise and factual. Do not mention "the context" or "the excerpts" — just answer.
"""

USER_PROMPT = """Context excerpts from the eBook:

{context}

Question: {question}
"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as labelled blocks so the model can cite pages."""
    return "\n\n".join(f"[Chunk {c.id} | Page {c.page}]\n{c.text}" for c in chunks)


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[tuple[str, str]]:
    return [
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT.format(context=format_context(chunks), question=question)),
    ]
