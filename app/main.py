import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.search import router as search_router
from app.services import embedding, qdrant, query_generation, reranker, sparse_embedding

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
if settings.debug:
    logging.getLogger("app").setLevel(logging.DEBUG)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting agentic retrieval service")
    log.info("Qdrant: uri=%s index=%s", settings.qdrant_uri, settings.qdrant_index)
    log.info(
        "Embedding: model=%s query_prefix=%r",
        settings.embedding_model,
        settings.embedding_prefix_query,
    )
    log.info(
        "Hybrid search: enable=%s sparse_provider=%s sparse_model=%s",
        settings.enable_hybrid_search,
        settings.sparse_query_provider,
        settings.sparse_query_model,
    )
    log.info("Reranking: %s", settings.enable_reranking)
    log.info("Agentic RAG: %s", settings.enable_agentic_rag)
    if settings.enable_agentic_rag:
        log.info("Agent model: %s", settings.agent_model)
        log.info("Agent API base: %s", settings.agent_api_base_url)
        log.info("Agent max iterations: %d", settings.agent_max_iterations)

    # Eagerly initialize Qdrant client
    qdrant.get_client()

    # Detect sparse-vector capability on the configured collection. Cached
    # for the process lifetime once a definite answer is obtained; if the
    # collection doesn't yet exist (cold-start before any ingest), we
    # re-detect on first query.
    if qdrant.collection_exists():
        sparse_present = qdrant.has_sparse_vectors()
        log.info(
            "Sparse vectors on collection %r: %s",
            settings.qdrant_index,
            sparse_present,
        )
        if settings.enable_hybrid_search and sparse_present:
            sparse_embedding.preload()
    else:
        log.warning(
            "Qdrant collection %r does not exist yet; sparse capability "
            "will be detected on first query.",
            settings.qdrant_index,
        )

    yield

    await embedding.close_client()
    await reranker.close_client()
    await query_generation.close_client()
    sparse_embedding.close()
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


@app.get("/health/ready")
async def health_ready():
    """Readiness probe — verifies Qdrant connectivity."""
    try:
        qdrant.get_client().get_collections()
        return {"status": "ok"}
    except Exception as exc:
        log.warning("Readiness check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(exc)},
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
