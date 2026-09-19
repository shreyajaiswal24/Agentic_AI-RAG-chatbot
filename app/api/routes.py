"""HTTP endpoints. Thin by design: validate, invoke the graph, map to the response model."""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.api.schemas import AskRequest, AskResponse, ChunkOut, ConfidenceOut, HealthResponse
from app.config import Settings
from app.errors import GenerationError, RetrievalError

logger = logging.getLogger(__name__)
router = APIRouter()


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@router.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Minimal browser chat UI (app/static/index.html); the JSON API is at /ask and /docs."""
    return FileResponse(STATIC_DIR / "index.html")


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        index=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        llm_provider=settings.llm_provider,
        llm_model=settings.llm_model,
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    graph = request.app.state.graph
    try:
        state = graph.invoke({"question": payload.question.strip()})
    except RetrievalError as exc:
        logger.exception("retrieval failed")
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {exc}") from exc
    except GenerationError as exc:
        logger.exception("generation failed")
        raise HTTPException(status_code=502, detail=f"Generation failed: {exc}") from exc

    return AskResponse(
        answer=state["answer"],
        grounded=state["grounded"],
        route=state["route"],
        relevance_score=state["relevance_score"],
        confidence=ConfidenceOut(**vars(state["confidence"])),
        retrieved_chunks=[
            ChunkOut(chunk_id=c.id, page=c.page, source=c.source, score=round(c.score, 4), text=c.text)
            for c in state["retrieved_chunks"]
        ],
    )
