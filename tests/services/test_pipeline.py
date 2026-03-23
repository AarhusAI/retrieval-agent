from unittest.mock import AsyncMock, patch

from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services.pipeline import extract_queries_from_messages, linear_search, search
from app.services.qdrant import QdrantResult

EMBED_PATH = "app.services.pipeline.embedding.embed_queries"
VSEARCH_PATH = "app.services.pipeline.qdrant.vector_search"
RERANK_PATH = "app.services.pipeline.rerank"


def _patch_embed(return_value):
    return patch(EMBED_PATH, new_callable=AsyncMock, return_value=return_value)


def _patch_vsearch(return_value):
    return patch(VSEARCH_PATH, new_callable=AsyncMock, return_value=return_value)


async def test_basic_search_flow():
    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=2)
    mock_qdrant_result = QdrantResult(
        texts=["doc1", "doc2"],
        metadatas=[{"src": "a"}, {"src": "b"}],
        distances=[0.9, 0.8],
    )

    with _patch_embed([[0.1, 0.2, 0.3]]), _patch_vsearch(mock_qdrant_result):
        result = await linear_search(request)

    assert isinstance(result, SearchResponse)
    assert len(result.documents) == 1
    assert result.documents[0] == ["doc1", "doc2"]


async def test_deduplication():
    request = SearchRequest(queries=["hello"], collection_names=["c1", "c2"], k=5)
    mock_result = QdrantResult(
        texts=["same text"],
        metadatas=[{"src": "a"}],
        distances=[0.9],
    )

    with _patch_embed([[0.1, 0.2]]), _patch_vsearch(mock_result):
        result = await linear_search(request)

    # "same text" appears in both collections but should be deduped to 1
    assert len(result.documents[0]) == 1


async def test_k_limits_results():
    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=1)
    mock_result = QdrantResult(
        texts=["doc1", "doc2", "doc3"],
        metadatas=[{}, {}, {}],
        distances=[0.9, 0.8, 0.7],
    )

    with _patch_embed([[0.1]]), _patch_vsearch(mock_result):
        result = await linear_search(request)

    assert len(result.documents[0]) == 1


async def test_multiple_queries():
    request = SearchRequest(queries=["q1", "q2"], collection_names=["coll1"], k=2)
    mock_result = QdrantResult(
        texts=["doc1"],
        metadatas=[{}],
        distances=[0.9],
    )

    with _patch_embed([[0.1], [0.2]]), _patch_vsearch(mock_result):
        result = await linear_search(request)

    assert len(result.documents) == 2


async def test_reranking_enabled(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.settings.enable_reranking", True)
    monkeypatch.setattr("app.services.pipeline.settings.initial_retrieval_multiplier", 3)

    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=2)
    mock_result = QdrantResult(
        texts=["doc1", "doc2"],
        metadatas=[{"a": 1}, {"b": 2}],
        distances=[0.9, 0.8],
    )
    rerank_return = (
        ["doc2", "doc1"],
        [{"b": 2}, {"a": 1}],
        [0.95, 0.7],
    )

    with (
        _patch_embed([[0.1]]),
        _patch_vsearch(mock_result) as mock_vs,
        patch(RERANK_PATH, new_callable=AsyncMock, return_value=rerank_return),
    ):
        result = await linear_search(request)

    # vector_search should be called with k * multiplier
    assert mock_vs.call_args[0][2] == 6
    assert result.documents[0] == ["doc2", "doc1"]


async def test_search_with_messages_extracts_query():
    """When only messages are provided, last user message is used as query."""
    request = SearchRequest(
        messages=[
            ChatMessage(role="user", content="first question"),
            ChatMessage(role="assistant", content="answer"),
            ChatMessage(role="user", content="search for this"),
        ],
        collection_names=["coll1"],
        k=2,
    )
    mock_result = QdrantResult(
        texts=["doc1"],
        metadatas=[{"src": "a"}],
        distances=[0.9],
    )

    with _patch_embed([[0.1, 0.2]]) as mock_embed, _patch_vsearch(mock_result):
        result = await linear_search(request)

    # Should have extracted "search for this" as the query
    mock_embed.assert_called_once_with(["search for this"])
    assert len(result.documents) == 1


async def test_search_with_queries_and_messages_prefers_queries():
    """When both queries and messages are provided, queries take precedence."""
    request = SearchRequest(
        queries=["explicit query"],
        messages=[ChatMessage(role="user", content="message query")],
        collection_names=["coll1"],
        k=2,
    )
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with _patch_embed([[0.1]]) as mock_embed, _patch_vsearch(mock_result):
        await linear_search(request)

    mock_embed.assert_called_once_with(["explicit query"])


def test_extract_queries_from_messages_last_user():
    messages = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="reply"),
        ChatMessage(role="user", content="second"),
    ]
    assert extract_queries_from_messages(messages) == ["second"]


def test_extract_queries_from_messages_no_user():
    messages = [ChatMessage(role="assistant", content="reply")]
    assert extract_queries_from_messages(messages) == []


def test_extract_queries_from_messages_empty():
    assert extract_queries_from_messages([]) == []


def test_extract_queries_from_messages_skips_empty_content():
    messages = [
        ChatMessage(role="user", content="real query"),
        ChatMessage(role="user", content="   "),
    ]
    assert extract_queries_from_messages(messages) == ["real query"]


async def test_search_routes_to_linear_when_agentic_disabled(monkeypatch):
    """When enable_agentic_rag=False, search() uses linear pipeline."""
    monkeypatch.setattr("app.services.pipeline.settings.enable_agentic_rag", False)

    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=2)
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with _patch_embed([[0.1]]), _patch_vsearch(mock_result):
        result = await search(request)

    assert isinstance(result, SearchResponse)
    assert result.documents[0] == ["doc1"]


async def test_search_routes_to_agentic_when_enabled(monkeypatch):
    """When enable_agentic_rag=True, search() delegates to agentic_search."""
    monkeypatch.setattr("app.services.pipeline.settings.enable_agentic_rag", True)

    mock_response = SearchResponse(
        documents=[["agentic-doc"]],
        metadatas=[[{"src": "agent"}]],
        distances=[[0.95]],
    )
    with patch(
        "app.services.agent.agentic_search",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=2)
        result = await search(request)

    assert result.documents[0] == ["agentic-doc"]
