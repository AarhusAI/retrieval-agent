import asyncio
import hashlib
import logging

from app.config import settings
from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services import embedding, qdrant
from app.services.bm25 import bm25_search, reciprocal_rank_fusion
from app.services.reranker import rerank

log = logging.getLogger(__name__)


def extract_queries_from_messages(messages: list[ChatMessage]) -> list[str]:
    """
    Extract search queries from chat messages.
    Uses the last user message content as the query.
    """
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return [message.content.strip()]
    return []


async def linear_search(request: SearchRequest) -> SearchResponse:
    """
    Traditional linear search pipeline:
      1. Resolve queries (from explicit queries or extracted from messages)
      2. Embed queries
      3. For each (query, collection): vector search
      4. Optional: BM25 hybrid fusion
      5. Optional: cross-encoder reranking
      6. Dedup, sort, limit to k
    """
    k = request.k
    fetch_k = k
    if settings.enable_reranking:
        fetch_k = k * settings.initial_retrieval_multiplier

    # Step 0: resolve queries — explicit queries take precedence over messages
    queries = request.queries
    if not queries and request.messages:
        queries = extract_queries_from_messages(request.messages)
        log.info("Extracted queries from %d messages: %s", len(request.messages), queries)

    if not queries:
        return SearchResponse(documents=[], metadatas=[], distances=[])

    # Step 1: embed all queries
    vectors = await embedding.embed_queries(queries)

    all_documents: list[list[str]] = []
    all_metadatas: list[list[dict]] = []
    all_distances: list[list[float]] = []

    # Step 2: search each query across all collections (concurrently)
    for _query_idx, (query_text, query_vector) in enumerate(
        zip(queries, vectors, strict=True)
    ):
        # Vector search across all collections concurrently
        tasks = [
            qdrant.vector_search(coll, query_vector, fetch_k) for coll in request.collection_names
        ]
        results = await asyncio.gather(*tasks)

        # Merge results from all collections for this query
        merged_texts: list[str] = []
        merged_metadatas: list[dict] = []
        merged_distances: list[float] = []

        for result in results:
            merged_texts.extend(result.texts)
            merged_metadatas.extend(result.metadatas)
            merged_distances.extend(result.distances)

        # Step 3: Optional hybrid search with BM25
        if settings.enable_hybrid_search and merged_texts:
            vector_ranked = list(zip(merged_texts, merged_distances, strict=True))
            bm25_tasks = [
                bm25_search(coll, query_text, fetch_k) for coll in request.collection_names
            ]
            bm25_results_per_coll = await asyncio.gather(*bm25_tasks)
            bm25_merged: list[tuple[str, float]] = []
            for bm25_res in bm25_results_per_coll:
                bm25_merged.extend(bm25_res)

            fused = reciprocal_rank_fusion(vector_ranked, bm25_merged, settings.hybrid_bm25_weight)

            # Rebuild ordered results from fused ranking
            text_to_meta = dict(zip(merged_texts, merged_metadatas, strict=True))
            merged_texts = [text for text, _ in fused]
            merged_distances = [score for _, score in fused]
            merged_metadatas = [text_to_meta.get(t, {}) for t in merged_texts]

        # Step 4: Optional reranking
        if settings.enable_reranking and merged_texts:
            merged_texts, merged_metadatas, merged_distances = await rerank(
                query_text, merged_texts, merged_metadatas, k
            )

        # Step 5: Dedup by text content hash, limit to k
        seen: set[str] = set()
        deduped_texts: list[str] = []
        deduped_metadatas: list[dict] = []
        deduped_distances: list[float] = []

        for text, meta, dist in zip(merged_texts, merged_metadatas, merged_distances, strict=True):
            text_hash = hashlib.md5(text.encode()).hexdigest()
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


async def search(request: SearchRequest) -> SearchResponse:
    """
    Main entry point: routes to agentic or linear pipeline based on config.
    """
    if settings.enable_agentic_rag:
        from app.services.agent import agentic_search

        log.info("Using agentic search pipeline")
        return await agentic_search(request)

    log.info("Using linear search pipeline")
    return await linear_search(request)
