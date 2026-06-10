from unittest.mock import AsyncMock, patch

import httpx

from app.config import settings
from app.services.reranker import rerank

_FAKE_REQUEST = httpx.Request("POST", "http://fake/v1/rerank")


async def test_rerank_basic_replace_mode():
    """RERANK_FUSION=replace ⇒ pure cross-encoder ordering (the older behaviour)."""
    mock_response = httpx.Response(
        200,
        json={
            "results": [
                {"index": 0, "relevance_score": 0.3},
                {"index": 1, "relevance_score": 0.9},
            ]
        },
        request=_FAKE_REQUEST,
    )
    with (
        patch.object(settings, "rerank_fusion", "replace"),
        patch.object(
            httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
        ),
    ):
        texts, metas, scores = await rerank(
            "query",
            ["doc_low", "doc_high"],
            [{"id": 1}, {"id": 2}],
            k=2,
        )

    # Should be sorted by relevance descending
    assert texts == ["doc_high", "doc_low"]
    assert metas == [{"id": 2}, {"id": 1}]
    assert scores == [0.9, 0.3]


def _diagnostic_response():
    """Reranker response mirroring the real 'Opkrævningens bankoplysninger' ranks.

    Documents are passed in retrieval (dense) order: index 0 is the answer chunk
    (dense #0) which the cross-encoder buries (rerank #3 @ 0.0776); two workflow
    chunks get the top cross-encoder scores.
    """
    return httpx.Response(
        200,
        json={
            "results": [
                {"index": 0, "relevance_score": 0.0776},  # bankopl  — dense#0
                {"index": 1, "relevance_score": 0.2434},  # C        — dense#1
                {"index": 2, "relevance_score": 0.8825},  # A        — dense#2
                {"index": 3, "relevance_score": 0.8018},  # B        — dense#3
            ]
        },
        request=_FAKE_REQUEST,
    )


_DIAG_DOCS = ["bankopl", "C", "A", "B"]
_DIAG_METAS = [{"id": 0}, {"id": 1}, {"id": 2}, {"id": 3}]


async def test_rerank_rrf_surfaces_buried_dense_hit():
    """Fix 4: RRF lifts the dense-#0 / rerank-#3 answer chunk into the top-3."""
    with (
        patch.object(settings, "rerank_fusion", "rrf"),
        patch.object(settings, "rerank_rrf_k", 60),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=_diagnostic_response(),
        ),
    ):
        texts, _metas, _scores = await rerank("query", _DIAG_DOCS, _DIAG_METAS, k=3)

    # Fused order (K=60): A (#0), bankopl (#2), C — bankopl is inside top-3.
    assert "bankopl" in texts
    assert texts == ["A", "bankopl", "C"]


async def test_rerank_replace_mode_buries_dense_hit():
    """Without fusion the cross-encoder drops the answer chunk out of top-3."""
    with (
        patch.object(settings, "rerank_fusion", "replace"),
        patch.object(
            httpx.AsyncClient,
            "post",
            new_callable=AsyncMock,
            return_value=_diagnostic_response(),
        ),
    ):
        texts, _metas, _scores = await rerank("query", _DIAG_DOCS, _DIAG_METAS, k=3)

    assert "bankopl" not in texts
    assert texts == ["A", "B", "C"]


async def test_rerank_limits_to_k():
    mock_response = httpx.Response(
        200,
        json={
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.8},
                {"index": 2, "relevance_score": 0.7},
            ]
        },
        request=_FAKE_REQUEST,
    )
    with patch.object(
        httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        texts, _metas, _scores = await rerank(
            "query",
            ["a", "b", "c"],
            [{}, {}, {}],
            k=2,
        )

    assert len(texts) == 2
    assert texts == ["a", "b"]


async def test_rerank_empty_documents():
    texts, metas, scores = await rerank("query", [], [], k=5)
    assert texts == []
    assert metas == []
    assert scores == []


async def test_rerank_http_error_falls_back_to_unranked():
    """When reranker API returns 500, fall back to unranked results truncated to k."""
    mock_response = httpx.Response(
        500,
        json={"error": "server error"},
        request=_FAKE_REQUEST,
    )
    with patch.object(
        httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
    ):
        texts, metas, scores = await rerank(
            "query",
            ["doc1", "doc2", "doc3"],
            [{"id": 1}, {"id": 2}, {"id": 3}],
            k=2,
        )

    # Should return first k documents unranked
    assert texts == ["doc1", "doc2"]
    assert metas == [{"id": 1}, {"id": 2}]
    assert scores == [0.0, 0.0]


async def test_rerank_connect_error_falls_back_to_unranked():
    """When reranker API is unreachable, fall back to unranked results."""
    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        texts, metas, scores = await rerank(
            "query",
            ["doc1", "doc2"],
            [{"id": 1}, {"id": 2}],
            k=5,
        )

    assert texts == ["doc1", "doc2"]
    assert metas == [{"id": 1}, {"id": 2}]
    assert scores == [0.0, 0.0]


async def test_rerank_failure_increments_failure_counter():
    """The fail-open path must bump reranker_failures_total."""
    from prometheus_client import REGISTRY

    before = REGISTRY.get_sample_value("reranker_failures_total") or 0.0
    with patch.object(
        httpx.AsyncClient,
        "post",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("Connection refused"),
    ):
        await rerank("query", ["doc1"], [{"id": 1}], k=5)
    after = REGISTRY.get_sample_value("reranker_failures_total")
    assert after == before + 1
