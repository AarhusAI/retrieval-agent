from unittest.mock import AsyncMock, patch

from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services.agent import (
    RetrievalResult,
    _dedup_results,
    agentic_search,
    extract_queries_from_messages,
)


class TestExtractQueries:
    def test_last_user_message(self):
        messages = [
            ChatMessage(role="user", content="first"),
            ChatMessage(role="assistant", content="reply"),
            ChatMessage(role="user", content="second"),
        ]
        assert extract_queries_from_messages(messages) == ["second"]

    def test_no_user_messages(self):
        messages = [ChatMessage(role="assistant", content="reply")]
        assert extract_queries_from_messages(messages) == []

    def test_empty(self):
        assert extract_queries_from_messages([]) == []

    def test_skips_empty_content(self):
        messages = [
            ChatMessage(role="user", content="real query"),
            ChatMessage(role="user", content="   "),
        ]
        assert extract_queries_from_messages(messages) == ["real query"]


class TestDedupResults:
    def test_basic_dedup(self):
        results = [
            RetrievalResult(texts=["a", "b"], metadatas=[{}, {}], distances=[0.9, 0.8]),
            RetrievalResult(texts=["a", "c"], metadatas=[{}, {}], distances=[0.95, 0.7]),
        ]
        texts, metas, dists = _dedup_results(results, k=10)
        assert texts == ["a", "b", "c"]

    def test_respects_k_limit(self):
        results = [
            RetrievalResult(
                texts=["a", "b", "c"], metadatas=[{}, {}, {}], distances=[0.9, 0.8, 0.7]
            ),
        ]
        texts, _, _ = _dedup_results(results, k=2)
        assert len(texts) == 2

    def test_empty_results(self):
        texts, metas, dists = _dedup_results([], k=5)
        assert texts == []
        assert metas == []
        assert dists == []


class TestAgenticSearch:
    async def test_empty_queries(self):
        request = SearchRequest(
            messages=[ChatMessage(role="assistant", content="no user msg")],
            collection_names=["coll1"],
        )
        result = await agentic_search(request)
        assert result.documents == []

    async def test_calls_agent_and_returns_results(self):
        """Mock the PydanticAI agent to verify the full flow."""
        mock_result = [
            RetrievalResult(
                texts=["doc1", "doc2"],
                metadatas=[{"src": "a"}, {"src": "b"}],
                distances=[0.9, 0.8],
            )
        ]

        mock_agent_result = AsyncMock()
        mock_agent_result.data = mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=5)
            result = await agentic_search(request)

        assert isinstance(result, SearchResponse)
        assert result.documents == [["doc1", "doc2"]]
        assert result.metadatas == [[{"src": "a"}, {"src": "b"}]]
        assert result.distances == [[0.9, 0.8]]

    async def test_deduplicates_across_retrieval_results(self):
        """Agent returns multiple retrieval results with overlapping docs."""
        mock_result = [
            RetrievalResult(texts=["same"], metadatas=[{"a": 1}], distances=[0.9]),
            RetrievalResult(texts=["same", "unique"], metadatas=[{"a": 1}, {"b": 2}], distances=[0.95, 0.8]),
        ]

        mock_agent_result = AsyncMock()
        mock_agent_result.data = mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=5)
            result = await agentic_search(request)

        assert result.documents == [["same", "unique"]]

    async def test_messages_included_in_agent_prompt(self):
        """When messages are provided alongside queries, they are passed to the agent as context."""
        mock_result = [
            RetrievalResult(texts=["doc1"], metadatas=[{}], distances=[0.9])
        ]

        mock_agent_result = AsyncMock()
        mock_agent_result.data = mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(
                queries=["Hvad mere kan du sige om dette"],
                messages=[
                    ChatMessage(role="user", content="Hvad kan elektronisk underskrift bruges til"),
                    ChatMessage(role="user", content="Hvad mere kan du sige om dette"),
                ],
                collection_names=["coll1"],
                k=3,
            )
            await agentic_search(request)

        # Verify the agent prompt includes conversation context
        prompt = mock_agent.run.call_args[0][0]
        assert "Conversation context:" in prompt
        assert "elektronisk underskrift" in prompt
        assert "Hvad mere kan du sige om dette" in prompt
