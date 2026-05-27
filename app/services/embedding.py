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


async def embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed query texts via OpenAI-compatible API. Applies query prefix before embedding."""
    prefixed = [f"{settings.embedding_prefix_query}{q}" for q in queries]

    url = f"{settings.embedding_api_base_url.rstrip('/')}/embeddings"
    payload = {"model": settings.embedding_model, "input": prefixed}
    headers = {}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    client = get_client()
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    # Sort by index to preserve order
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]
