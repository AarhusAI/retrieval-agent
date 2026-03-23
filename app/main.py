import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routes.search import router as search_router
from app.services import qdrant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting agentic retrieval service")
    log.info("Qdrant URI: %s", settings.qdrant_uri)
    log.info("Multitenancy: %s", settings.qdrant_multitenancy)
    log.info("Embedding model: %s", settings.embedding_model)
    log.info("Hybrid search: %s", settings.enable_hybrid_search)
    log.info("Reranking: %s", settings.enable_reranking)
    log.info("Agentic RAG: %s", settings.enable_agentic_rag)
    if settings.enable_agentic_rag:
        log.info("Agent model: %s", settings.agent_model)
        log.info("Agent API base: %s", settings.agent_api_base_url)
        log.info("Agent max iterations: %d", settings.agent_max_iterations)

    # Eagerly initialize Qdrant client
    qdrant.get_client()

    yield

    qdrant.close_client()
    log.info("Agentic retrieval service shut down")


app = FastAPI(
    title="Agentic Retrieval Service",
    description="Agentic retrieval engine for Open WebUI",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(search_router)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
