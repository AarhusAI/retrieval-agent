from unittest.mock import AsyncMock, MagicMock, patch

from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services.agent import (
    AgentDeps,
    RetrievalResult,
    _dedup_results,
    agentic_search,
    extract_queries_from_messages,
)


def _mock_usage(input_tokens=100, output_tokens=50, requests=2, tool_calls=1):
    """Create a mock PydanticAI Usage object."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.requests = requests
    usage.tool_calls = tool_calls
    return usage


def _make_mock_agent(full_results: list[RetrievalResult]):
    """Create a mock agent whose run() populates deps.full_results."""
    mock_agent_result = MagicMock()
    mock_agent_result.output = "done"
    mock_agent_result.usage.return_value = _mock_usage()

    async def _run(prompt, *, deps: AgentDeps, **kwargs):
        deps.full_results = full_results
        return mock_agent_result

    mock_agent = AsyncMock()
    mock_agent.run = AsyncMock(side_effect=_run)
    return mock_agent


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
        texts, _metas, _dists = _dedup_results(results, k=10)
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

        mock_agent = _make_mock_agent(mock_result)

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
            RetrievalResult(
                texts=["same", "unique"],
                metadatas=[{"a": 1}, {"b": 2}],
                distances=[0.95, 0.8],
            ),
        ]

        mock_agent = _make_mock_agent(mock_result)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=5)
            result = await agentic_search(request)

        assert result.documents == [["same", "unique"]]

    async def test_messages_used_as_primary_input(self):
        """When messages are present, conversation is the primary input."""
        mock_result = [RetrievalResult(texts=["doc1"], metadatas=[{}], distances=[0.9])]

        mock_agent = _make_mock_agent(mock_result)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(
                queries=["Hvad mere kan du sige om dette"],
                messages=[
                    ChatMessage(
                        role="user",
                        content="Hvad kan elektronisk underskrift bruges til",
                    ),
                    ChatMessage(role="user", content="Hvad mere kan du sige om dette"),
                ],
                collection_names=["coll1"],
                k=3,
            )
            await agentic_search(request)

        # Verify the agent prompt uses conversation as primary input
        prompt = mock_agent.run.call_args[0][0]
        assert "Conversation:" in prompt
        assert "elektronisk underskrift" in prompt
        assert "Hvad mere kan du sige om dette" in prompt
        assert "Today's date:" in prompt
