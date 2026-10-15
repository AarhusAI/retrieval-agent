import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app import http_client
from app.auth import verify_api_key
from app.config import settings
from app.logging_config import configure_logging
from app.routes.search import router as search_router
from app.services import qdrant, query_generation, sparse_embedding

configure_logging(settings)
log = logging.getLogger(__name__)

# Settings safe to print at startup. Never add *_key or prompt fields here.
_LOGGED_SETTINGS = {
    "log_level",
    "log_level_app",
    "log_format",
    "metrics_enabled",
    "qdrant_uri",
    "qdrant_index",
    "embedding_model",
    "embedding_prefix_query",
    "enable_hybrid_search",
    "sparse_query_provider",
    "sparse_query_model",
    "enable_reranking",
    "enable_agentic_rag",
    "agent_model",
    "agent_api_base_url",
    "agent_max_iterations",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting agentic retrieval service")
    # Allowlist (not an exclude list) so a new credential field can never leak
    # into the startup log by omission.
    log.info("Config: %s", settings.model_dump(include=_LOGGED_SETTINGS))

    # Eagerly initialize Qdrant client
    qdrant.get_client()

    # Detect sparse-vector capability on the configured collection. Cached
    # for the process lifetime once a definite answer is obtained; if the
    # collection doesn't yet exist (cold-start before any ingest), this is
    # False (uncached) and we re-detect on first query.
    sparse_present = qdrant.has_sparse_vectors()
    log.info("Sparse vectors on collection %r: %s", settings.qdrant_index, sparse_present)
    if settings.enable_hybrid_search and sparse_present:
        sparse_embedding.get_model()

    yield

    await http_client.close_client()
    await query_generation.close_client()
    # The agent module is imported lazily (pipeline only pulls in pydantic_ai
    # when agentic mode runs) — close its transport only if it was ever loaded,
    # without forcing the import here.
    agent_module = sys.modules.get("app.services.agent")
    if agent_module is not None:
        await agent_module.close_client()
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


@app.get("/metrics", dependencies=[Depends(verify_api_key)])
async def metrics():
    """Prometheus scrape endpoint.

    Bearer-authenticated with the same ``API_KEY`` as ``/search`` — the scrape
    job must send ``Authorization: Bearer <API_KEY>``. Returns 404 when
    METRICS_ENABLED=false; instrumentation runs regardless.
    """
    if not settings.metrics_enabled:
        return JSONResponse(status_code=404, content={"detail": "metrics disabled"})
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health/ready")
async def health_ready():
    """Readiness probe — verifies Qdrant connectivity."""
    try:
        # Sync Qdrant client — probe in a worker thread so a hung Qdrant
        # connection can't stall the event loop (and with it /health).
        await asyncio.to_thread(lambda: qdrant.get_client().get_collections())
        return {"status": "ok"}
    except Exception as exc:
        # Log full exception server-side; surface only a generic status to
        # callers to avoid leaking internal hostnames/ports.
        log.warning("Readiness check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "error"},
        )
