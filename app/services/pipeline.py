"""Linear search pipeline + the entrypoint that routes to linear/agentic.

Phase 3 simplification: ``vector_search`` now hits a single physical Qdrant
collection (filtered by ``meta.collection_name IN (collection_names)``), so
the per-collection ``asyncio.gather`` is gone — one query, one result set.

When ``ENABLE_HYBRID_SEARCH`` is on:

  * If the target collection has sparse vectors, the sparse query embedder
    runs and Qdrant fuses dense + sparse server-side via RRF — no client-side
    BM25 path executes.
  * If the collection has no sparse vectors, BM25 RRF runs client-side over
    the same collection scope.
"""

import hashlib
import logging
import time

from qdrant_client.http.models import SparseVector

from app import metrics
from app.config import settings
from app.log_utils import sanitize_for_log
from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services import embedding, qdrant, sparse_embedding
from app.services.bm25 import bm25_search, reciprocal_rank_fusion
from app.services.reranker import rerank

log = logging.getLogger(__name__)


def extract_queries_from_messages(messages: list[ChatMessage]) -> list[str]:
    """Use the last non-empty user message as the query."""
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return [message.content.strip()]
    return []


async def embed_dense_and_sparse(
    queries: list[str],
) -> tuple[list[list[float]], list[SparseVector | None], bool]:
    """Embed queries for retrieval. Returns ``(dense, sparse, use_native_hybrid)``.

    Probes :func:`qdrant.has_sparse_vectors` once. Sparse embedding only runs
    when hybrid is enabled *and* the configured collection actually carries a
    ``text-sparse`` named vector; otherwise the sparse list is all-``None``
    and downstream callers fall back to dense-only retrieval (with optional
    client-side BM25 RRF if hybrid is on).
    """
    async with metrics.time_stage("embed_dense"):
        vectors = await embedding.embed_queries(queries)
    use_native_hybrid = settings.enable_hybrid_search and qdrant.has_sparse_vectors()
    if use_native_hybrid:
        async with metrics.time_stage("embed_sparse"):
            sparse_vectors: list[SparseVector | None] = await sparse_embedding.embed_queries(
                queries
            )
    else:
        sparse_vectors = [None] * len(queries)

    # Record which fusion path this search will take so operators can see
    # whether native server-side RRF, the client-side BM25 fallback, or
    # dense-only retrieval is actually running.
    if use_native_hybrid:
        path = "native_sparse"
    elif settings.enable_hybrid_search:
        path = "bm25_fallback"
    else:
        path = "dense_only"
    metrics.hybrid_path_total.labels(path=path).inc()

    return vectors, sparse_vectors, use_native_hybrid


async def retrieve_one_query(
    query_text: str,
    query_vector: list[float],
    sparse_vec: SparseVector | None,
    collection_names: list[str],
    fetch_k: int,
    use_native_hybrid: bool,
    *,
    rerank_k: int | None,
) -> tuple[list[str], list[dict], list[float]]:
    """Run vector search + optional client-side BM25 RRF + optional rerank.

    ``rerank_k=None`` skips reranking even when ``ENABLE_RERANKING`` is true
    (used by the agent fallback path, which intentionally returns raw vector
    results). Returns parallel lists ``(texts, metadatas, distances)``.

    Client-side BM25 RRF only runs when ``ENABLE_HYBRID_SEARCH`` is on AND
    ``use_native_hybrid`` is false — native hybrid already RRF-fuses
    server-side, so client-side fusion would double-rank.
    """
    async with metrics.time_stage("qdrant"):
        result = await qdrant.vector_search(
            collection_names,
            query_vector,
            sparse_vec,
            fetch_k,
        )
    texts = list(result.texts)
    metadatas = list(result.metadatas)
    distances = list(result.distances)
    metrics.candidates_fetched.observe(len(texts))
    log.debug(
        "retrieve_one_query q=%s candidates=%d top_scores=%s",
        sanitize_for_log(query_text),
        len(texts),
        [round(d, 4) for d in distances[:3]],
    )

    if settings.enable_hybrid_search and not use_native_hybrid and texts:
        async with metrics.time_stage("bm25"):
            vector_ranked = list(zip(texts, distances, strict=True))
            bm25_results = await bm25_search(collection_names, query_text, fetch_k)
            fused = reciprocal_rank_fusion(
                vector_ranked, bm25_results, settings.hybrid_bm25_weight
            )
            text_to_meta: dict[str, dict] = {}
            for text, meta in zip(texts, metadatas, strict=True):
                if text not in text_to_meta:
                    text_to_meta[text] = meta
            texts = [text for text, _ in fused]
            distances = [score for _, score in fused]
            metadatas = [text_to_meta.get(t, {}) for t in texts]

    if settings.enable_reranking and texts and rerank_k is not None:
        async with metrics.time_stage("rerank"):
            texts, metadatas, distances = await rerank(query_text, texts, metadatas, rerank_k)

    return texts, metadatas, distances


async def linear_search(request: SearchRequest) -> SearchResponse:
    """Traditional linear pipeline:

    1. Resolve queries (LLM generation → explicit ``queries`` → last user message)
    2. Embed (dense, plus sparse when hybrid+sparse-capable)
    3. Single Qdrant query per dense query, scoped to all collection_names
    4. Optional client-side BM25 RRF (only when hybrid is on AND no sparse)
    5. Optional cross-encoder reranking
    6. Dedup by MD5, limit to k
    """
    k = request.k
    fetch_k = k * settings.initial_retrieval_multiplier if settings.enable_reranking else k

    queries = await _resolve_queries(request)
    if not queries:
        return SearchResponse(documents=[], metadatas=[], distances=[])

    vectors, sparse_vectors, use_native_hybrid = await embed_dense_and_sparse(queries)

    all_documents: list[list[str]] = []
    all_metadatas: list[list[dict]] = []
    all_distances: list[list[float]] = []

    for query_text, query_vector, sparse_vec in zip(queries, vectors, sparse_vectors, strict=True):
        merged_texts, merged_metadatas, merged_distances = await retrieve_one_query(
            query_text,
            query_vector,
            sparse_vec,
            request.collection_names,
            fetch_k,
            use_native_hybrid,
            rerank_k=k,
        )

        # Dedup by text content hash, limit to k
        seen: set[str] = set()
        deduped_texts: list[str] = []
        deduped_metadatas: list[dict] = []
        deduped_distances: list[float] = []

        for text, meta, dist in zip(merged_texts, merged_metadatas, merged_distances, strict=True):
            text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
            if text_hash in seen:
                continue
            seen.add(text_hash)
            deduped_texts.append(text)
            deduped_metadatas.append(meta)
            deduped_distances.append(dist)
            if len(deduped_texts) >= k:
                break

        all_documents.append(deduped_texts)
        all_metadatas.append(deduped_metadatas)
        all_distances.append(deduped_distances)

    return SearchResponse(
        documents=all_documents,
        metadatas=all_metadatas,
        distances=all_distances,
    )


async def _resolve_queries(request: SearchRequest) -> list[str]:
    """Pick query strings from request, preferring LLM generation when available."""
    queries: list[str] | None = None

    if request.messages and settings.enable_query_generation:
        from app.services.query_generation import generate_queries_from_messages

        async with metrics.time_stage("query_generation"):
            queries = await generate_queries_from_messages(
                request.messages,
                template_override=request.retrieval_query_generation_prompt_template,
            )
        if queries:
            log.info("Generated %d queries from messages", len(queries))
            log.debug("Generated queries: %s", sanitize_for_log(queries))

    if not queries:
        queries = request.queries or []

    if not queries and request.messages:
        queries = extract_queries_from_messages(request.messages)
        log.info("Extracted %d queries from %d messages", len(queries), len(request.messages))
        log.debug("Extracted queries: %s", sanitize_for_log(queries))

    return queries


async def search(request: SearchRequest) -> SearchResponse:
    """Main entry point: routes to agentic or linear pipeline based on config.

    Single owner of request-level metrics: ``search_requests_total`` (success
    and error), ``search_duration_seconds`` and ``results_returned`` are all
    recorded here so there's no double-counting and the ``pipeline`` label is
    consistent across both.
    """
    pipeline = "agentic" if settings.enable_agentic_rag else "linear"
    start = time.perf_counter()
    try:
        if settings.enable_agentic_rag:
            from app.services.agent import agentic_search

            log.info("Using agentic search pipeline")
            response = await agentic_search(request)
        else:
            log.info("Using linear search pipeline")
            response = await linear_search(request)
    except Exception as exc:
        metrics.search_requests_total.labels(
            pipeline=pipeline, outcome="error", code=type(exc).__name__
        ).inc()
        raise
    else:
        metrics.search_requests_total.labels(
            pipeline=pipeline, outcome="success", code="none"
        ).inc()
        metrics.results_returned.observe(sum(len(d) for d in response.documents))
        return response
    finally:
        metrics.search_duration_seconds.labels(pipeline=pipeline).observe(
            time.perf_counter() - start
        )
