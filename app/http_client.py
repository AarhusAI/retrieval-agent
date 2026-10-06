"""Shared httpx client for the embedding and reranker APIs."""

import httpx

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


def bearer(api_key: str) -> dict[str, str]:
    """Authorization header for ``api_key``; empty when no key is configured."""
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}
