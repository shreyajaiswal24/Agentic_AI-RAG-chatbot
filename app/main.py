"""FastAPI application factory.

Expensive objects (embedding client, Pinecone index handle, LLM, compiled
graph) are built once in the lifespan and shared via ``app.state``.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.graph.builder import build_graph
from app.graph.nodes import RAGNodes
from app.llm.client import get_chat_model
from app.llm.grader import LLMGrader
from app.retrieval.embeddings import get_embeddings
from app.retrieval.vector_store import PineconeVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def build_dependencies(settings: Settings):
    """Compose the real graph. Separated so tests can inject a fake one."""
    store = PineconeVectorStore(settings, get_embeddings(settings))
    store.stats()  # fail fast at startup if the index/namespace is not reachable
    llm = get_chat_model(settings)
    nodes = RAGNodes(store, llm, LLMGrader(llm), top_k=settings.top_k, relevance_threshold=settings.relevance_threshold)
    return build_graph(nodes)


def create_app(graph=None, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = settings or get_settings()
        app.state.graph = graph or build_dependencies(app.state.settings)
        logger.info("RAG graph ready (index=%s, llm=%s/%s)", app.state.settings.pinecone_index_name,
                    app.state.settings.llm_provider, app.state.settings.llm_model)
        yield

    app = FastAPI(
        title="Agentic AI eBook RAG Chatbot",
        description="Answers questions strictly from Konverge AI's 'Agentic AI' eBook using LangGraph + Pinecone.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
