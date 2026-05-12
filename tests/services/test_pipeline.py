from unittest.mock import AsyncMock, patch

from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services.pipeline import extract_queries_from_messages, linear_search, search
from app.services.qdrant import QdrantResult

EMBED_PATH = "app.services.pipeline.embedding.embed_queries"
VSEARCH_PATH = "app.services.pipeline.qdrant.vector_search"
HAS_SPARSE_PATH = "app.services.pipeline.qdrant.has_sparse_vectors"
SPARSE_EMBED_PATH = "app.services.pipeline.sparse_embedding.embed_queries"
RERANK_PATH = "app.services.pipeline.rerank"


def _patch_embed(return_value):
    return patch(EMBED_PATH, new_callable=AsyncMock, return_value=return_value)


def _patch_vsearch(return_value):
    return patch(VSEARCH_PATH, new_callable=AsyncMock, return_value=return_value)


def _patch_has_sparse(return_value=False):
    return patch(HAS_SPARSE_PATH, return_value=return_value)


async def test_basic_search_flow():
    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=2)
    mock_qdrant_result = QdrantResult(
        texts=["doc1", "doc2"],
        metadatas=[{"src": "a"}, {"src": "b"}],
        distances=[0.9, 0.8],
    )

    with _patch_embed([[0.1, 0.2, 0.3]]), _patch_vsearch(mock_qdrant_result), _patch_has_sparse():
        result = await linear_search(request)

    assert isinstance(result, SearchResponse)
    assert len(result.documents) == 1
    assert result.documents[0] == ["doc1", "doc2"]


async def test_vector_search_called_with_collection_names_list():
    """Pipeline passes the full collection_names list as a single arg — no per-collection gather."""  # noqa: E501
    request = SearchRequest(queries=["hello"], collection_names=["c1", "c2"], k=2)
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with (
        _patch_embed([[0.1]]),
        _patch_vsearch(mock_result) as mock_vs,
        _patch_has_sparse(),
    ):
        await linear_search(request)

    # Single call (one query, one collection-set), 4-arg signature.
    assert mock_vs.call_count == 1
    args, _ = mock_vs.call_args
    assert args[0] == ["c1", "c2"]
    assert args[1] == [0.1]
    assert args[2] is None  # sparse vector
    assert args[3] == 2  # k


async def test_deduplication():
    """Identical chunks deduped on the way out."""
    request = SearchRequest(queries=["hello"], collection_names=["c1"], k=5)
    mock_result = QdrantResult(
        texts=["same text", "same text", "other"],
        metadatas=[{"src": "a"}, {"src": "b"}, {"src": "c"}],
        distances=[0.9, 0.85, 0.8],
    )

    with _patch_embed([[0.1, 0.2]]), _patch_vsearch(mock_result), _patch_has_sparse():
        result = await linear_search(request)

    # "same text" appears twice but should be deduped to 1
    assert result.documents[0] == ["same text", "other"]


async def test_k_limits_results():
    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=1)
    mock_result = QdrantResult(
        texts=["doc1", "doc2", "doc3"],
        metadatas=[{}, {}, {}],
        distances=[0.9, 0.8, 0.7],
    )

    with _patch_embed([[0.1]]), _patch_vsearch(mock_result), _patch_has_sparse():
        result = await linear_search(request)

    assert len(result.documents[0]) == 1


async def test_multiple_queries():
    request = SearchRequest(queries=["q1", "q2"], collection_names=["coll1"], k=2)
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with _patch_embed([[0.1], [0.2]]), _patch_vsearch(mock_result), _patch_has_sparse():
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
        _patch_has_sparse(),
        patch(RERANK_PATH, new_callable=AsyncMock, return_value=rerank_return),
    ):
        result = await linear_search(request)

    # vector_search should be called with k * multiplier as the 4th positional arg
    assert mock_vs.call_args.args[3] == 6
    assert result.documents[0] == ["doc2", "doc1"]


async def test_search_with_messages_extracts_query_when_generation_disabled(monkeypatch):
    """When query generation is disabled and only messages provided, last user message is used."""
    monkeypatch.setattr("app.services.pipeline.settings.enable_query_generation", False)

    request = SearchRequest(
        messages=[
            ChatMessage(role="user", content="first question"),
            ChatMessage(role="assistant", content="answer"),
            ChatMessage(role="user", content="search for this"),
        ],
        collection_names=["coll1"],
        k=2,
    )
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{"src": "a"}], distances=[0.9])

    with (
        _patch_embed([[0.1, 0.2]]) as mock_embed,
        _patch_vsearch(mock_result),
        _patch_has_sparse(),
    ):
        result = await linear_search(request)

    mock_embed.assert_called_once_with(["search for this"])
    assert len(result.documents) == 1


async def test_search_with_messages_uses_query_generation(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.settings.enable_query_generation", True)

    request = SearchRequest(
        queries=["raw last message"],
        messages=[ChatMessage(role="user", content="raw last message")],
        collection_names=["coll1"],
        k=2,
    )
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with (
        _patch_embed([[0.1]]) as mock_embed,
        _patch_vsearch(mock_result),
        _patch_has_sparse(),
        patch(
            "app.services.query_generation.generate_queries_from_messages",
            new_callable=AsyncMock,
            return_value=["optimized query"],
        ),
    ):
        await linear_search(request)

    mock_embed.assert_called_once_with(["optimized query"])


async def test_search_falls_back_to_queries_when_generation_fails(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.settings.enable_query_generation", True)

    request = SearchRequest(
        queries=["explicit query"],
        messages=[ChatMessage(role="user", content="message query")],
        collection_names=["coll1"],
        k=2,
    )
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with (
        _patch_embed([[0.1]]) as mock_embed,
        _patch_vsearch(mock_result),
        _patch_has_sparse(),
        patch(
            "app.services.query_generation.generate_queries_from_messages",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        await linear_search(request)

    mock_embed.assert_called_once_with(["explicit query"])


async def test_search_with_queries_only_no_generation(monkeypatch):
    monkeypatch.setattr("app.services.pipeline.settings.enable_query_generation", True)

    request = SearchRequest(
        queries=["explicit query"],
        collection_names=["coll1"],
        k=2,
    )
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with (
        _patch_embed([[0.1]]) as mock_embed,
        _patch_vsearch(mock_result),
        _patch_has_sparse(),
    ):
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
    monkeypatch.setattr("app.services.pipeline.settings.enable_agentic_rag", False)

    request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=2)
    mock_result = QdrantResult(texts=["doc1"], metadatas=[{}], distances=[0.9])

    with _patch_embed([[0.1]]), _patch_vsearch(mock_result), _patch_has_sparse():
        result = await search(request)

    assert isinstance(result, SearchResponse)
    assert result.documents[0] == ["doc1"]


async def test_bm25_fallback_runs_when_hybrid_on_and_no_sparse(monkeypatch):
    """Hybrid + no sparse vectors → client-side BM25 RRF fires; sparse embedder doesn't."""
    monkeypatch.setattr("app.services.pipeline.settings.enable_hybrid_search", True)

    request = SearchRequest(queries=["hello"], collection_names=["c1", "c2"], k=5)
    mock_result = QdrantResult(texts=["same text"], metadatas=[{"src": "first"}], distances=[0.9])

    bm25_call = AsyncMock(return_value=[("same text", 5.0)])
    sparse_call = AsyncMock(return_value=[None])

    with (
        _patch_embed([[0.1]]),
        _patch_vsearch(mock_result),
        _patch_has_sparse(False),
        patch("app.services.pipeline.bm25_search", bm25_call),
        patch(SPARSE_EMBED_PATH, sparse_call),
    ):
        result = await linear_search(request)

    bm25_call.assert_called_once()
    args, _ = bm25_call.call_args
    assert args[0] == ["c1", "c2"]
    sparse_call.assert_not_called()
    assert result.metadatas[0][0]["src"] == "first"


async def test_native_hybrid_skips_client_bm25(monkeypatch):
    """Hybrid + sparse vectors present → sparse embedder runs, client-side BM25 does NOT."""
    monkeypatch.setattr("app.services.pipeline.settings.enable_hybrid_search", True)

    request = SearchRequest(queries=["hello"], collection_names=["c1"], k=2)
    mock_result = QdrantResult(texts=["doc"], metadatas=[{}], distances=[0.5])

    bm25_call = AsyncMock(return_value=[])
    sparse_call = AsyncMock(return_value=[None])  # sparse_query_provider=none case

    with (
        _patch_embed([[0.1]]),
        _patch_vsearch(mock_result) as mock_vs,
        _patch_has_sparse(True),
        patch("app.services.pipeline.bm25_search", bm25_call),
        patch(SPARSE_EMBED_PATH, sparse_call),
    ):
        await linear_search(request)

    bm25_call.assert_not_called()
    sparse_call.assert_called_once()
    # vector_search received the sparse vector slot (None when provider=none)
    args, _ = mock_vs.call_args
    assert args[2] is None


async def test_search_routes_to_agentic_when_enabled(monkeypatch):
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
