"""Answer grading (LLM-as-judge) — the source of the API's confidence score.

After ``generate`` produces an answer, a second, independent LLM call reads
the *same* retrieved excerpts and fact-checks the answer against them,
claim by claim. The confidence score is computed in code as
``supported_claims / total_claims`` so it is an evidence-support ratio, not a
self-reported feeling. This is separate from ``relevance_score`` (retrieval
similarity): a question can retrieve very similar text and still get an
answer the text does not support, or vice versa.
"""

from typing import Literal, Protocol

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from app.errors import GenerationError, describe_exception
from app.llm.prompts import format_context
from app.retrieval.models import RetrievedChunk

Verdict = Literal["fully_supported", "partially_supported", "not_supported"]


class AnswerGrade(BaseModel):
    """Structured output the judge model must return."""

    total_claims: int = Field(ge=0, description="Number of distinct factual claims in the answer")
    supported_claims: int = Field(ge=0, description="How many of those claims are explicitly supported by the excerpts")
    verdict: Verdict
    reason: str = Field(description="One or two sentences naming any unsupported claim")


class Grader(Protocol):
    def grade(self, question: str, answer: str, chunks: list[RetrievedChunk]) -> AnswerGrade: ...


GRADER_SYSTEM_PROMPT = """You are a strict fact-checker for a question-answering system.
You will receive excerpts from a document, a question, and an answer that was supposed to be \
written ONLY from those excerpts.

Your job:
1. Split the answer into its distinct factual claims. Page citations such as "(Page 19)" are NOT claims: \
strip them and never judge whether a citation points at the right page.
2. For each claim, decide whether it is supported by ANY of the excerpts, taken together — not only the \
excerpt on the cited page. Paraphrase counts as support; information that is merely plausible, from \
general knowledge, or absent from all excerpts does NOT.
3. Report total_claims, supported_claims, a verdict, and a short reason naming any unsupported claim.

verdict rules: fully_supported if every claim is supported; not_supported if none are; \
partially_supported otherwise. Be strict and literal."""

GRADER_USER_PROMPT = """Excerpts:

{context}

Question: {question}

Answer to check: {answer}
"""


class LLMGrader:
    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm.with_structured_output(AnswerGrade)

    def grade(self, question: str, answer: str, chunks: list[RetrievedChunk]) -> AnswerGrade:
        messages = [
            ("system", GRADER_SYSTEM_PROMPT),
            ("human", GRADER_USER_PROMPT.format(context=format_context(chunks), question=question, answer=answer)),
        ]
        try:
            grade = self._llm.invoke(messages)
        except Exception as exc:
            raise GenerationError(f"Answer grading failed: {describe_exception(exc)}") from exc
        if not isinstance(grade, AnswerGrade):  # structured output returned None / raw text
            raise GenerationError("Answer grading returned no structured result")
        return grade
