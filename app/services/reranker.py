import logging

import httpx

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
    Returns (texts, metadatas, scores) sorted by relevance score, limited to k.
    Falls back to unranked results (truncated to k) on HTTP or connection errors.
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
    except (httpx.HTTPStatusError, httpx.ConnectError) as exc:
        log.warning("Reranker request failed (%s), returning unranked results", exc)
        return documents[:k], metadatas[:k], [0.0] * min(len(documents), k)

    # Response format: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
    results = sorted(data["results"], key=lambda x: x["index"])
    scores = [r["relevance_score"] for r in results]

    scored = list(zip(documents, metadatas, scores, strict=True))
    scored.sort(key=lambda x: x[2], reverse=True)
    scored = scored[:k]

    texts = [s[0] for s in scored]
    metas = [s[1] for s in scored]
    dists = [s[2] for s in scored]
    return texts, metas, dists
