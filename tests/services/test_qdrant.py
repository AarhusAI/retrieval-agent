from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.http.exceptions import UnexpectedResponse

from app.services.qdrant import QdrantResult, vector_search


def _make_point(id, text, metadata, score):
    point = MagicMock()
    point.id = id
    point.payload = {"text": text, "metadata": metadata}
    point.score = score
    return point


async def test_vector_search_basic():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [
        _make_point("1", "hello world", {"source": "a"}, 0.8),
        _make_point("2", "foo bar", {"source": "b"}, 0.6),
    ]
    mock_client.query_points.return_value = mock_response

    with patch("app.services.qdrant.get_client", return_value=mock_client):
        result = await vector_search("file-abc", [0.1, 0.2], k=5)

    assert isinstance(result, QdrantResult)
    assert result.texts == ["hello world", "foo bar"]
    assert result.metadatas == [{"source": "a"}, {"source": "b"}]
    # Cosine normalization: (score + 1) / 2
    assert result.distances[0] == (0.8 + 1.0) / 2.0
    assert result.distances[1] == (0.6 + 1.0) / 2.0


async def test_vector_search_empty_payload():
    mock_client = MagicMock()
    mock_response = MagicMock()
    point = MagicMock()
    point.payload = None
    point.score = 0.5
    mock_response.points = [point]
    mock_client.query_points.return_value = mock_response

    with patch("app.services.qdrant.get_client", return_value=mock_client):
        result = await vector_search("file-abc", [0.1], k=5)

    assert result.texts == [""]
    assert result.metadatas == [{}]


async def test_vector_search_unexpected_response_returns_empty():
    """UnexpectedResponse (e.g. collection not found) returns empty results."""
    mock_client = MagicMock()
    mock_client.query_points.side_effect = UnexpectedResponse(
        status_code=404, content=b"Not found", reason_phrase="Not Found", headers={}
    )

    with patch("app.services.qdrant.get_client", return_value=mock_client):
        result = await vector_search("file-abc", [0.1], k=5)

    assert result.texts == []
    assert result.metadatas == []
    assert result.distances == []


async def test_vector_search_connection_error_propagates():
    """Connection errors (not UnexpectedResponse) should propagate."""
    mock_client = MagicMock()
    mock_client.query_points.side_effect = ConnectionError("connection refused")

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        pytest.raises(ConnectionError),
    ):
        await vector_search("file-abc", [0.1], k=5)


async def test_vector_search_applies_tenant_filter():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    with patch("app.services.qdrant.get_client", return_value=mock_client):
        await vector_search("file-abc", [0.1], k=5)

    call_kwargs = mock_client.query_points.call_args[1]
    assert call_kwargs["query_filter"] is not None
