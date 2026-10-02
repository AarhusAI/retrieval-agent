import logging

from app.config import settings
from app.http_client import bearer, get_client

log = logging.getLogger(__name__)


async def embed_queries(queries: list[str]) -> list[list[float]]:
    """Embed query texts via OpenAI-compatible API. Applies query prefix before embedding."""
    if not queries:
        # OpenAI-compatible APIs reject an empty input list.
        return []

    prefixed = [f"{settings.embedding_prefix_query}{q}" for q in queries]

    url = f"{settings.embedding_api_base_url.rstrip('/')}/embeddings"
    payload = {"model": settings.embedding_model, "input": prefixed}
    resp = await get_client().post(url, json=payload, headers=bearer(settings.embedding_api_key))
    resp.raise_for_status()
    data = resp.json()

    try:
        # Sort by index to preserve order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]
    except (KeyError, TypeError) as exc:
        # Loud on purpose (like Qdrant errors) but with a clear message instead
        # of a bare KeyError bubbling out of the response dict.
        raise ValueError(f"Unexpected embedding API response shape: {exc!r}") from exc
