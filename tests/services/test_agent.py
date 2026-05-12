from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services.agent import (
    AgentDeps,
    RetrievalResult,
    _build_previews,
    _dedup_results,
    _parse_fallback_queries,
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

    async def test_fallback_handles_embed_failure(self):
        """When agent doesn't call retrieve and embed_queries fails, returns empty results."""
        mock_agent_result = MagicMock()
        mock_agent_result.output = "I could not process that"
        mock_agent_result.usage.return_value = _mock_usage()
        mock_agent_result.all_messages.return_value = []

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        with (
            patch("app.services.agent._get_agent", return_value=mock_agent),
            patch(
                "app.services.pipeline.embedding.embed_queries",
                new_callable=AsyncMock,
                side_effect=RuntimeError("Embedding API down"),
            ),
        ):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=5)
            result = await agentic_search(request)

        assert isinstance(result, SearchResponse)
        assert result.documents == [[]]

    async def test_retrieve_accumulates_results_across_retries(self):
        """When agent calls retrieve twice, results from both calls are accumulated."""
        first_results = [
            RetrievalResult(texts=["doc1"], metadatas=[{"src": "a"}], distances=[0.9])
        ]
        second_results = [
            RetrievalResult(texts=["doc2"], metadatas=[{"src": "b"}], distances=[0.8])
        ]

        call_count = 0

        async def _run(prompt, *, deps: AgentDeps, **kwargs):
            nonlocal call_count
            # Simulate two retrieve calls by building up full_results
            deps.full_results = (deps.full_results or []) + first_results
            deps.full_results = (deps.full_results or []) + second_results
            call_count += 1
            mock_result = MagicMock()
            mock_result.output = "done"
            mock_result.usage.return_value = _mock_usage()
            mock_result.all_messages.return_value = []
            return mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=_run)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=10)
            result = await agentic_search(request)

        assert "doc1" in result.documents[0]
        assert "doc2" in result.documents[0]


class TestBuildPreviews:
    def test_caps_at_preview_k(self):
        """Preview output is bounded by preview_k regardless of input size."""
        all_results = [
            RetrievalResult(
                texts=[f"doc{i}" for i in range(15)],
                metadatas=[{"source": f"s{i}"} for i in range(15)],
                distances=[1.0 - i * 0.01 for i in range(15)],
            )
        ]
        previews = _build_previews(all_results, max_chars=200, preview_k=5)
        assert len(previews) == 1
        assert len(previews[0].texts) == 5
        assert previews[0].texts == ["doc0", "doc1", "doc2", "doc3", "doc4"]

    def test_dedups_across_queries_within_preview_k(self):
        """Duplicates across queries don't consume preview_k slots twice."""
        all_results = [
            RetrievalResult(texts=["a", "b"], metadatas=[{}, {}], distances=[0.9, 0.8]),
            RetrievalResult(texts=["a", "c"], metadatas=[{}, {}], distances=[0.95, 0.7]),
        ]
        previews = _build_previews(all_results, max_chars=200, preview_k=10)
        assert previews[0].texts == ["a", "b", "c"]

    def test_truncates_long_text(self):
        all_results = [
            RetrievalResult(texts=["x" * 500], metadatas=[{"source": "s"}], distances=[0.9])
        ]
        previews = _build_previews(all_results, max_chars=50, preview_k=5)
        assert previews[0].texts[0] == "x" * 50 + "..."

    def test_metadata_carries_structural_fields(self):
        """Previews carry source + page + headers + collection_type as structured
        fields so the retrieval-specialist agent can grade relevance against
        section/page signals. Non-whitelisted fields stay in deps.full_results.
        """
        all_results = [
            RetrievalResult(
                texts=["t"],
                metadatas=[
                    {
                        "source": "doc.pdf",
                        "page": 7,
                        "headers": ["Privacy", "Foundational Principles"],
                        "collection_type": "file",
                        # Anything outside the whitelist is dropped from previews.
                        "score": 0.99,
                        "file_id": "abc-123",
                        "user_id": "u-1",
                    }
                ],
                distances=[0.9],
            )
        ]
        previews = _build_previews(all_results, max_chars=200, preview_k=5)
        assert previews[0].metadatas == [
            {
                "source": "doc.pdf",
                "page": 7,
                "headers": ["Privacy", "Foundational Principles"],
                "collection_type": "file",
            }
        ]

    def test_metadata_omits_missing_optional_fields(self):
        """Optional fields are dropped when absent, ``None``, ``""``, or ``[]`` —
        the agent shouldn't burn context on ``"headers": []`` noise. Numeric
        zero is kept (it's a valid value; our writers shouldn't emit it for
        page indices, but if they do we want it visible)."""
        all_results = [
            RetrievalResult(
                texts=["t1", "t2", "t3"],
                metadatas=[
                    # Plain-text mode chunk — no structural fields.
                    {"source": "a.pdf"},
                    # Explicit empty / None — should be filtered.
                    {"source": "b.md", "headers": [], "page": None},
                    # Mixed: collection_type kept, headers/page absent → dropped.
                    {"source": "c.pdf", "collection_type": "memory"},
                ],
                distances=[0.9, 0.8, 0.7],
            )
        ]
        previews = _build_previews(all_results, max_chars=200, preview_k=5)
        assert previews[0].metadatas == [
            {"source": "a.pdf"},
            {"source": "b.md"},
            {"source": "c.pdf", "collection_type": "memory"},
        ]

    def test_metadata_keeps_source_even_when_blank(self):
        """source is the only field every preview must carry — matches the
        previous contract so consumers can rely on ``meta["source"]`` existing."""
        all_results = [RetrievalResult(texts=["t"], metadatas=[{}], distances=[0.9])]
        previews = _build_previews(all_results, max_chars=200, preview_k=5)
        assert previews[0].metadatas == [{"source": ""}]


class TestAgentRecallDecoupling:
    async def test_fetch_k_uses_agent_fetch_k_not_request_k(self):
        """The agent's candidate pool is agent_fetch_k, not request.k.

        Open WebUI sends k=3, but the agent should still grade against a
        wide candidate pool so the relevant chunk isn't filtered out before
        the LLM ever sees it.
        """
        captured_deps: list[AgentDeps] = []

        async def _run(prompt, *, deps: AgentDeps, **kwargs):
            captured_deps.append(deps)
            deps.full_results = [RetrievalResult(texts=["doc"], metadatas=[{}], distances=[0.9])]
            mock_result = MagicMock()
            mock_result.output = "done"
            mock_result.usage.return_value = _mock_usage()
            mock_result.all_messages.return_value = []
            return mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=_run)

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=3)
            await agentic_search(request)

        assert len(captured_deps) == 1
        assert captured_deps[0].k == 3
        assert captured_deps[0].fetch_k == settings.agent_fetch_k
        assert captured_deps[0].fetch_k > 3

    async def test_conversation_history_is_capped(self):
        """Only the last N messages reach the agent's user prompt."""
        mock_result = [RetrievalResult(texts=["doc"], metadatas=[{}], distances=[0.9])]
        mock_agent = _make_mock_agent(mock_result)

        n = settings.agent_conversation_history_messages
        old_messages = [ChatMessage(role="user", content=f"OLD_MSG_{i}") for i in range(10)]
        recent_messages = [ChatMessage(role="user", content=f"RECENT_MSG_{i}") for i in range(n)]

        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(
                messages=old_messages + recent_messages,
                collection_names=["coll1"],
                k=3,
            )
            await agentic_search(request)

        prompt = mock_agent.run.call_args[0][0]
        for i in range(10):
            assert f"OLD_MSG_{i}" not in prompt
        for i in range(n):
            assert f"RECENT_MSG_{i}" in prompt


class TestParseFallbackQueries:
    def test_mistral_tool_calls_with_trailing_text(self):
        """Greedy regex fix: trailing text after JSON should not break parsing."""
        output = '[TOOL_CALLS]retrieve{"queries": ["test query"]} some trailing text'
        result = _parse_fallback_queries(output)
        assert result == ["test query"]

    def test_plain_json(self):
        output = '{"queries": ["q1", "q2"]}'
        result = _parse_fallback_queries(output)
        assert result == ["q1", "q2"]

    def test_no_queries(self):
        output = "I cannot help with that."
        result = _parse_fallback_queries(output)
        assert result is None
