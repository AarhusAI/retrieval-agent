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

    def test_interleaves_so_second_query_survives_small_k(self):
        """A unique, lower-scored doc from the second query must survive a small k.

        Regression: the first query's results used to fill k entirely and later
        queries were never considered, dropping a relevant-but-low-scored chunk
        (e.g. a Danish chunk the cross-encoder under-ranked). Round-robin gives
        the second query's best hit a slot near the front instead.
        """
        results = [
            # Query 1: high-scored but off-target chunks (e.g. table-of-contents).
            RetrievalResult(
                texts=["toc1", "toc2", "toc3"],
                metadatas=[{}, {}, {}],
                distances=[0.80, 0.76, 0.70],
            ),
            # Query 2: the actually-relevant chunk, under-scored by the reranker.
            RetrievalResult(
                texts=["answer", "noise1", "noise2"],
                metadatas=[{}, {}, {}],
                distances=[0.13, 0.09, 0.09],
            ),
        ]
        texts, _metas, dists = _dedup_results(results, k=3)
        assert "answer" in texts
        assert texts == ["toc1", "answer", "toc2"]
        assert dists == [0.80, 0.13, 0.76]


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

    async def test_seeds_raw_query_before_agent_loop(self):
        """The user's original query is retrieved and merged even when the
        agent's own queries drift off-target (Fix 2a)."""
        # Agent contributes a high-scored but off-target chunk, appending like
        # the real retrieve tool does.
        drift = RetrievalResult(texts=["toc"], metadatas=[{"source": "toc"}], distances=[0.9])

        async def _run(prompt, *, deps: AgentDeps, **kwargs):
            deps.full_results = (deps.full_results or []) + [drift]
            result = MagicMock()
            result.output = "done"
            result.usage.return_value = _mock_usage()
            result.all_messages.return_value = []
            return result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=_run)

        async def _embed(queries):
            return ([[0.1]] * len(queries), [None] * len(queries), False)

        async def _retrieve_one(query_text, *args, **kwargs):
            return (["ANSWER"], [{"source": "Bankoplysninger.pdf"}], [0.06])

        with (
            patch("app.services.agent._get_agent", return_value=mock_agent),
            patch.object(settings, "agent_include_raw_query", True),
            patch("app.services.agent.embed_dense_and_sparse", new=AsyncMock(side_effect=_embed)),
            patch(
                "app.services.agent.retrieve_one_query", new=AsyncMock(side_effect=_retrieve_one)
            ),
        ):
            request = SearchRequest(
                messages=[ChatMessage(role="user", content="Kan du oplyse bankoplysninger?")],
                collection_names=["coll1"],
                k=3,
            )
            result = await agentic_search(request)

        flat = result.documents[0]
        assert "ANSWER" in flat  # seeded raw-query result survived the merge
        assert "toc" in flat  # the agent's own result is present too

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


class TestAgentRetryObservability:
    async def test_two_retrieve_rounds_count_as_retry(self):
        """retrieve → evaluate → retrieve in the message history means the agent
        judged the first round off-topic and searched again: that's a retry, and
        ``agent_retries_total`` must increment."""
        from prometheus_client import REGISTRY
        from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
        from pydantic_ai.usage import RequestUsage

        messages = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="retrieve", args={"queries": ["q1"]})],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            ),
            ModelResponse(
                parts=[TextPart(content="off-topic, retrying")],
                usage=RequestUsage(input_tokens=8, output_tokens=4),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name="retrieve", args={"queries": ["q2 better"]})],
                usage=RequestUsage(input_tokens=12, output_tokens=6),
            ),
        ]

        async def _run(prompt, *, deps: AgentDeps, **kwargs):
            deps.full_results = [RetrievalResult(texts=["doc"], metadatas=[{}], distances=[0.9])]
            mock_result = MagicMock()
            mock_result.output = "done"
            mock_result.usage.return_value = _mock_usage(requests=3)
            mock_result.all_messages.return_value = messages
            return mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=_run)

        before = REGISTRY.get_sample_value("agent_retries_total") or 0.0
        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=5)
            result = await agentic_search(request)

        after = REGISTRY.get_sample_value("agent_retries_total")
        assert after == before + 1
        assert result.documents == [["doc"]]

    async def test_single_retrieve_round_is_not_a_retry(self):
        from prometheus_client import REGISTRY
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.usage import RequestUsage

        messages = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="retrieve", args={"queries": ["q1"]})],
                usage=RequestUsage(input_tokens=10, output_tokens=5),
            ),
        ]

        async def _run(prompt, *, deps: AgentDeps, **kwargs):
            deps.full_results = [RetrievalResult(texts=["doc"], metadatas=[{}], distances=[0.9])]
            mock_result = MagicMock()
            mock_result.output = "done"
            mock_result.usage.return_value = _mock_usage(requests=2)
            mock_result.all_messages.return_value = messages
            return mock_result

        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(side_effect=_run)

        before = REGISTRY.get_sample_value("agent_retries_total") or 0.0
        with patch("app.services.agent._get_agent", return_value=mock_agent):
            request = SearchRequest(queries=["hello"], collection_names=["coll1"], k=5)
            await agentic_search(request)

        after = REGISTRY.get_sample_value("agent_retries_total") or 0.0
        assert after == before


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

    def test_interleaves_across_queries(self):
        """Previews round-robin across queries so the grader sees each query's best.

        Mirrors what _dedup_results returns in the final payload, so the grader's
        view and the returned documents stay consistent.
        """
        all_results = [
            RetrievalResult(
                texts=["a1", "a2", "a3"],
                metadatas=[{"source": "a"}, {"source": "a"}, {"source": "a"}],
                distances=[0.9, 0.8, 0.7],
            ),
            RetrievalResult(
                texts=["b1", "b2"],
                metadatas=[{"source": "b"}, {"source": "b"}],
                distances=[0.2, 0.1],
            ),
        ]
        previews = _build_previews(all_results, max_chars=200, preview_k=4)
        # pos0 -> a1, b1 ; pos1 -> a2, b2 ; capped at 4
        assert previews[0].texts == ["a1", "b1", "a2", "b2"]

    def test_truncates_long_text(self):
        all_results = [
            RetrievalResult(texts=["x" * 500], metadatas=[{"source": "s"}], distances=[0.9])
        ]
        previews = _build_previews(all_results, max_chars=50, preview_k=5)
        assert previews[0].texts[0] == "x" * 50 + "..."

    def test_metadata_carries_structural_fields(self):
        """Previews carry source + title + page + headers + languages +
        collection_type as structured fields so the retrieval-specialist agent
        can grade relevance against section/page/title/language signals.
        Non-whitelisted fields stay in deps.full_results.
        """
        all_results = [
            RetrievalResult(
                texts=["t"],
                metadatas=[
                    {
                        "source": "doc.pdf",
                        "title": "Privacy by Design",
                        "page": 7,
                        "headers": ["Privacy", "Foundational Principles"],
                        "languages": ["en", "da"],
                        "collection_type": "file",
                        # Anything outside the whitelist is dropped from previews.
                        "score": 0.99,
                        "file_id": "abc-123",
                        "user_id": "u-1",
                        "authors": ["Fred Carter"],  # in Qdrant payload, not in preview
                        "created_at": "2010-11-02T15:06:47Z",  # same
                    }
                ],
                distances=[0.9],
            )
        ]
        previews = _build_previews(all_results, max_chars=200, preview_k=5)
        assert previews[0].metadatas == [
            {
                "source": "doc.pdf",
                "title": "Privacy by Design",
                "page": 7,
                "headers": ["Privacy", "Foundational Principles"],
                "languages": ["en", "da"],
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
