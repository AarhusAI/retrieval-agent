import logging

import httpx

from app.config import settings

log = logging.getLogger(__name__)


async def embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed query texts via OpenAI-compatible API. Applies query prefix before embedding."""
    prefixed = [f"{settings.embedding_query_prefix}{q}" for q in queries]

    url = f"{settings.embedding_api_base_url.rstrip('/')}/embeddings"
    payload = {"model": settings.embedding_model, "input": prefixed}
    headers = {}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()

    # Sort by index to preserve order
    sorted_data = sorted(data["data"], key=lambda x: x["index"])
    return [item["embedding"] for item in sorted_data]
