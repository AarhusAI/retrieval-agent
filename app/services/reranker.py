import logging

import httpx

from app import metrics
from app.config import settings

log = logging.getLogger(__name__)

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=30.0)
    return _client


async def close_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


async def rerank(
    query: str,
    documents: list[str],
    metadatas: list[dict],
    k: int,
) -> tuple[list[str], list[dict], list[float]]:
    """
    Rerank documents via OpenAI-compatible /v1/rerank endpoint.

    ``documents`` must arrive in retrieval-score order (callers pass them straight
    from vector/hybrid search), so the input index *is* the retrieval rank. With
    ``RERANK_FUSION=rrf`` (default) the final order is a Reciprocal Rank Fusion of
    the retrieval ranking and the cross-encoder ranking — so a strong dense hit the
    cross-encoder underranks isn't buried. ``replace`` restores cross-encoder-only
    ordering. Either way the returned score is the cross-encoder relevance score
    (only the ordering changes), limited to k. Falls back to unranked results
    (truncated to k) on any HTTP/transport error (including timeouts) or a
    malformed response body.
    """
    if not documents:
        return [], [], []

    url = f"{settings.reranker_api_base_url.rstrip('/')}/v1/rerank"
    payload = {
        "model": settings.reranker_model,
        "query": query,
        "documents": documents,
        "top_n": len(documents),  # get scores for all, we sort ourselves
    }
    headers = {}
    if settings.reranker_api_key:
        headers["Authorization"] = f"Bearer {settings.reranker_api_key}"

    try:
        client = get_client()
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        # Response format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        results = sorted(data["results"], key=lambda x: x["index"])
        scores = [r["relevance_score"] for r in results]
        if len(scores) != len(documents):
            raise ValueError(
                f"reranker returned {len(scores)} scores for {len(documents)} documents"
            )
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        # httpx.HTTPError covers status, connect, timeout and protocol errors;
        # the rest classify a malformed response body. Either way: fail open.
        metrics.reranker_failures_total.inc()
        log.warning("Reranker request failed (%s), returning unranked results", exc)
        return documents[:k], metadatas[:k], [0.0] * min(len(documents), k)

    n = len(documents)

    if settings.rerank_fusion == "rrf":
        # Reciprocal Rank Fusion: input index = retrieval rank, cross-encoder
        # order = rerank rank. Fusing the two stops the cross-encoder from
        # burying a strong dense/hybrid hit it happens to underrank. Stable sort
        # ⇒ retrieval order breaks fused-score ties.
        rf = settings.rerank_rrf_k
        rerank_order = sorted(range(n), key=lambda i: scores[i], reverse=True)
        rerank_rank = {idx: pos for pos, idx in enumerate(rerank_order)}
        order = sorted(
            range(n),
            key=lambda i: 1.0 / (rf + i) + 1.0 / (rf + rerank_rank[i]),
            reverse=True,
        )[:k]
    else:  # "replace" — cross-encoder score only (older behaviour)
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]

    # Return the cross-encoder relevance score (meaning unchanged downstream);
    # only the ordering reflects the fusion.
    texts = [documents[i] for i in order]
    metas = [metadatas[i] for i in order]
    dists = [scores[i] for i in order]
    return texts, metas, dists
