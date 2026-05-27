"""vector_search tests against the Phase 3 schema (content/meta + collection_name filter)."""

from unittest.mock import MagicMock, patch

import pytest
from qdrant_client import models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.services.qdrant import QdrantResult, vector_search


def _make_point(id, content, meta, score):
    point = MagicMock()
    point.id = id
    point.payload = {"content": content, "meta": meta}
    point.score = score
    return point


async def test_vector_search_basic():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = [
        _make_point("1", "hello world", {"source": "a", "collection_name": "file-abc"}, 0.8),
        _make_point("2", "foo bar", {"source": "b", "collection_name": "file-abc"}, 0.6),
    ]
    mock_client.query_points.return_value = mock_response

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        patch("app.services.qdrant.has_sparse_vectors", return_value=False),
    ):
        result = await vector_search(["file-abc"], [0.1, 0.2], None, k=5)

    assert isinstance(result, QdrantResult)
    assert result.texts == ["hello world", "foo bar"]
    assert result.metadatas == [
        {"source": "a", "collection_name": "file-abc"},
        {"source": "b", "collection_name": "file-abc"},
    ]
    # Cosine normalization (dense path): (score + 1) / 2
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

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        patch("app.services.qdrant.has_sparse_vectors", return_value=False),
    ):
        result = await vector_search(["file-abc"], [0.1], None, k=5)

    assert result.texts == [""]
    assert result.metadatas == [{}]


async def test_vector_search_unexpected_response_returns_empty():
    """UnexpectedResponse (e.g. collection not found) returns empty results."""
    mock_client = MagicMock()
    mock_client.query_points.side_effect = UnexpectedResponse(
        status_code=404, content=b"Not found", reason_phrase="Not Found", headers={}
    )

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        patch("app.services.qdrant.has_sparse_vectors", return_value=False),
    ):
        result = await vector_search(["file-abc"], [0.1], None, k=5)

    assert result.texts == []
    assert result.metadatas == []
    assert result.distances == []


async def test_vector_search_connection_error_propagates():
    """Connection errors (not UnexpectedResponse) should propagate."""
    mock_client = MagicMock()
    mock_client.query_points.side_effect = ConnectionError("connection refused")

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        patch("app.services.qdrant.has_sparse_vectors", return_value=False),
        pytest.raises(ConnectionError),
    ):
        await vector_search(["file-abc"], [0.1], None, k=5)


async def test_vector_search_filter_targets_meta_collection_name():
    """Filter scopes to meta.collection_name IN (collection_names) — the new schema."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        patch("app.services.qdrant.has_sparse_vectors", return_value=False),
    ):
        await vector_search(["file-abc", "kb-xyz"], [0.1], None, k=5)

    call_kwargs = mock_client.query_points.call_args.kwargs
    qfilter = call_kwargs["query_filter"]
    assert isinstance(qfilter, models.Filter)
    assert len(qfilter.must) == 1
    cond = qfilter.must[0]
    assert cond.key == "meta.collection_name"
    assert cond.match.any == ["file-abc", "kb-xyz"]


async def test_vector_search_uses_named_dense_vector():
    """Dense-only path passes ``using=text-dense`` so qdrant-haystack vectors resolve."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    with (
        patch("app.services.qdrant.get_client", return_value=mock_client),
        patch("app.services.qdrant.has_sparse_vectors", return_value=False),
    ):
        await vector_search(["file-abc"], [0.1, 0.2], None, k=5)

    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "text-dense"
    assert call_kwargs["query"] == [0.1, 0.2]
    # Pure dense path — no prefetch
    assert "prefetch" not in call_kwargs or call_kwargs.get("prefetch") is None
