from unittest.mock import AsyncMock, patch

import httpx

from app.services.reranker import rerank

_FAKE_REQUEST = httpx.Request("POST", "http://fake/v1/rerank")


async def test_rerank_basic():
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
    with patch.object(
        httpx.AsyncClient, "post", new_callable=AsyncMock, return_value=mock_response
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
