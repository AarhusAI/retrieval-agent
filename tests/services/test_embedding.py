from unittest.mock import AsyncMock, patch

import httpx
import pytest

from app.config import settings
from app.services.embedding import embed_queries

_FAKE_REQUEST = httpx.Request("POST", "http://fake/embeddings")


async def test_embed_queries_single():
    mock_response = httpx.Response(
        200,
        json={"data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]},
        request=_FAKE_REQUEST,
    )
    with patch.object(
        httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        result = await embed_queries(["hello"])

    assert len(result) == 1
    assert result[0] == [0.1, 0.2, 0.3]


async def test_embed_queries_preserves_order():
    mock_response = httpx.Response(
        200,
        json={
            "data": [
                {"index": 1, "embedding": [0.4, 0.5]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        },
        request=_FAKE_REQUEST,
    )
    with patch.object(
        httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        result = await embed_queries(["first", "second"])

    assert result[0] == [0.1, 0.2]
    assert result[1] == [0.4, 0.5]


async def test_embed_queries_applies_prefix():
    mock_response = httpx.Response(
        200,
        json={"data": [{"index": 0, "embedding": [0.1]}]},
        request=_FAKE_REQUEST,
    )
    with patch.object(
        httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
    ) as mock_post:
        await embed_queries(["test query"])

    call_payload = mock_post.call_args[1]["json"]
    # Prefix comes from settings.embedding_prefix_query
    assert call_payload["input"][0].startswith(settings.embedding_prefix_query)
    assert "test query" in call_payload["input"][0]


async def test_embed_queries_empty_input_skips_api_call():
    """Empty input must short-circuit — OpenAI-compatible APIs reject input=[]."""
    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        result = await embed_queries([])

    mock_post.assert_not_called()
    assert result == []


async def test_embed_queries_malformed_response_raises_value_error():
    """Unexpected response shape raises a clear ValueError, not a bare KeyError."""
    mock_response = httpx.Response(
        200,
        json={"unexpected": "shape"},
        request=_FAKE_REQUEST,
    )
    with (
        patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ),
        pytest.raises(ValueError, match="embedding API response"),
    ):
        await embed_queries(["hello"])


async def test_embed_queries_http_error():
    mock_response = httpx.Response(
        500,
        json={"error": "server error"},
        request=_FAKE_REQUEST,
    )
    with (
        patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ),
        pytest.raises(httpx.HTTPStatusError),
    ):
        await embed_queries(["hello"])
