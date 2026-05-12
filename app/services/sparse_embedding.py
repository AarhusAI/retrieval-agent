"""Sparse query embedder for native Qdrant hybrid retrieval.

Uses ``fastembed`` in-process — no external API call, but the model weights
are downloaded on first construction. The :func:`preload` hook is called from
``app.main``'s lifespan when hybrid is enabled, so first chat query doesn't
pay the model-load cost.

Disabled when ``SPARSE_QUERY_PROVIDER=none`` (``embed_queries`` returns a list
of ``None`` of the same length as the input). The pipeline / agent layer
treats a ``None`` sparse vector as "use dense-only retrieval", so this is the
clean knob to keep hybrid semantics intact while turning off the sparse stage.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from qdrant_client.http.models import SparseVector

from app.config import settings

if TYPE_CHECKING:
    from fastembed import SparseTextEmbedding

log = logging.getLogger(__name__)


_model: SparseTextEmbedding | None = None


def get_model() -> SparseTextEmbedding | None:
    """Lazy-load the fastembed sparse model. ``None`` when the provider is disabled."""
    if settings.sparse_query_provider.lower() == "none":
        return None

    global _model
    if _model is None:
        from fastembed import SparseTextEmbedding

        log.info("Loading fastembed sparse model: %s", settings.sparse_query_model)
        _model = SparseTextEmbedding(model_name=settings.sparse_query_model)
        log.info("Loaded fastembed sparse model: %s", settings.sparse_query_model)
    return _model


def preload() -> None:
    """Eagerly construct the model. No-op when hybrid or the provider is disabled.

    Called from the FastAPI lifespan so the model download / import is done
    before any requests reach the service.
    """
    if not settings.enable_hybrid_search:
        return
    if settings.sparse_query_provider.lower() == "none":
        return
    get_model()


def close() -> None:
    """Drop the cached model. Test hook; in production the model lives until SIGTERM."""
    global _model
    _model = None


async def embed_queries(queries: list[str]) -> list[SparseVector | None]:
    """Embed query strings with the configured sparse model.

    Returns one entry per query — a :class:`SparseVector` when the model is
    active, or ``None`` when sparse is disabled. The downstream caller treats
    ``None`` as "no sparse vector available, use dense-only".

    fastembed is synchronous; the embedding work runs in a worker thread so it
    doesn't block the FastAPI event loop.
    """
    model = get_model()
    if model is None:
        return [None] * len(queries)

    raw = await asyncio.to_thread(lambda: list(model.embed(queries)))
    return [SparseVector(indices=e.indices.tolist(), values=e.values.tolist()) for e in raw]
