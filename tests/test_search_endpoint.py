from unittest.mock import AsyncMock, patch

from app.models import SearchResponse

SEARCH_PATH = "app.routes.search.search"


async def test_search_returns_pipeline_results(client, api_headers):
    mock_response = SearchResponse(
        documents=[["doc1", "doc2"]],
        metadatas=[[{"source": "a"}, {"source": "b"}]],
        distances=[[0.9, 0.8]],
    )
    with patch(SEARCH_PATH, new_callable=AsyncMock, return_value=mock_response):
        resp = await client.post(
            "/search",
            json={"queries": ["hello"], "collection_names": ["coll1"], "k": 5},
            headers=api_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents"] == [["doc1", "doc2"]]
    assert data["metadatas"] == [[{"source": "a"}, {"source": "b"}]]
    assert data["distances"] == [[0.9, 0.8]]


async def test_search_with_messages_only(client, api_headers):
    mock_response = SearchResponse(
        documents=[["doc1"]],
        metadatas=[[{"source": "a"}]],
        distances=[[0.9]],
    )
    with patch(SEARCH_PATH, new_callable=AsyncMock, return_value=mock_response):
        resp = await client.post(
            "/search",
            json={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi there"},
                    {"role": "user", "content": "search for this"},
                ],
                "collection_names": ["coll1"],
                "k": 5,
            },
            headers=api_headers,
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["documents"] == [["doc1"]]


async def test_search_with_both_queries_and_messages(client, api_headers):
    """When both queries and messages are provided, queries take precedence."""
    mock_response = SearchResponse(documents=[[]], metadatas=[[]], distances=[[]])
    with patch(SEARCH_PATH, new_callable=AsyncMock, return_value=mock_response) as mock:
        await client.post(
            "/search",
            json={
                "queries": ["explicit query"],
                "messages": [{"role": "user", "content": "message query"}],
                "collection_names": ["coll1"],
            },
            headers=api_headers,
        )
    # Verify the request was passed through with both fields
    req = mock.call_args[0][0]
    assert req.queries == ["explicit query"]
    assert req.messages is not None


async def test_search_missing_queries_and_messages(client, api_headers):
    """Neither queries nor messages -> 422."""
    resp = await client.post(
        "/search",
        json={"collection_names": ["coll1"]},
        headers=api_headers,
    )
    assert resp.status_code == 422


async def test_search_missing_collections(client, api_headers):
    resp = await client.post(
        "/search",
        json={"queries": ["hello"]},
        headers=api_headers,
    )
    assert resp.status_code == 422


async def test_search_k_must_be_positive(client, api_headers):
    resp = await client.post(
        "/search",
        json={"queries": ["hello"], "collection_names": ["coll1"], "k": 0},
        headers=api_headers,
    )
    assert resp.status_code == 422


async def test_search_default_k(client, api_headers):
    mock_response = SearchResponse(documents=[[]], metadatas=[[]], distances=[[]])
    with patch(SEARCH_PATH, new_callable=AsyncMock, return_value=mock_response) as mock:
        await client.post(
            "/search",
            json={"queries": ["hello"], "collection_names": ["coll1"]},
            headers=api_headers,
        )
    assert mock.call_args[0][0].k == 5
