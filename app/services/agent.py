"""
PydanticAI agent for agentic RAG — query analysis, rewriting, decomposition,
retrieval, and corrective relevance grading with retry.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from itertools import islice

import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from app import metrics
from app.config import settings
from app.log_utils import log_llm_request, sanitize_for_log
from app.models import RetrievalResult, SearchRequest, SearchResponse
from app.services.pipeline import (
    dedup_topk,
    embed_dense_and_sparse,
    extract_queries_from_messages,
    interleave_dedup,
    retrieve_one_query,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependencies injected into the agent via RunContext
# ---------------------------------------------------------------------------


@dataclass
class AgentDeps:
    """Dependencies available to the agent's tools."""

    collection_names: list[str]
    k: int
    fetch_k: int
    # Side-channel: full results stored here, truncated previews sent to LLM
    full_results: list[RetrievalResult] | None = None
    # Per-retrieve-round insight (queries built + per-query hit counts + top
    # scores), appended once per ``retrieve`` tool call. Drives the DEBUG
    # round-trace and makes the agent's retry behaviour observable.
    round_stats: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent definition
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a retrieval specialist for a RAG system. Your job is to find the most \
relevant documents for a user's information needs by searching a vector database.

## Query Analysis Guidelines
- Generate 1-2 search queries optimized for semantic vector search.
- Base queries on the **user's questions and information needs only**. \
Use assistant responses solely for context and disambiguation \
(e.g. resolving "this", "that", "the one you mentioned").
- Generate queries as natural-language phrases that capture the semantic \
meaning of the user's information need.
- Reformulate conversational references into standalone, self-contained queries.
- If the user's message clearly needs no document retrieval (e.g. greetings, \
small talk), call the retrieve tool with an empty list to signal no results needed.
- Respond in the same language as the user's messages.

## Retrieval Strategy
1. SEARCH — call the retrieve tool with your optimized queries.
2. ACCEPT the results if **any** returned document is on-topic for the user's \
question, even partially. Partial coverage is expected — the downstream LLM \
will synthesize the answer.
3. RETRY only if the results are **completely off-topic** (none of the returned \
documents relate to the query at all). Rewrite the query to be more specific \
and try again (up to {max_iterations} attempts total). Do not retry just \
because the answer is not explicitly stated — relevant context is enough.

When grading relevance, use any structural or descriptive metadata each \
result exposes — ``title`` (the document's own title), ``headers`` (the \
section/heading breadcrumb a chunk sits under), ``page`` (page number in \
the source document), and ``languages`` (detected language codes — useful \
to spot language mismatches between user query and result) are strong \
topical signals even when the chunk's body text is terse or generic.

Keep queries concise and focused. Prefer a single well-crafted query over \
multiple overlapping ones.\
"""


# Held at module level so the lifespan can close it at shutdown; the OpenAI
# SDK does not own a caller-supplied http_client.
_http_client: httpx.AsyncClient | None = None


def _build_agent() -> Agent[AgentDeps, str]:
    """Build the PydanticAI agent. Called once at module level."""
    global _http_client
    _http_client = httpx.AsyncClient(
        # Transport-level backstop aligned with the run-level wall clock
        # (asyncio.wait_for in agentic_search); the OpenAI SDK may override
        # per request, the run bound is what actually caps a hung call.
        timeout=httpx.Timeout(settings.agent_timeout),
        # Request hook logs the faithful LLM payload at DEBUG (app.llm).
        event_hooks={"request": [log_llm_request]},
    )
    model = OpenAIChatModel(
        settings.agent_model,
        provider=OpenAIProvider(
            base_url=settings.agent_api_base_url or None,
            api_key=settings.agent_api_key or None,
            http_client=_http_client,
        ),
        profile=OpenAIModelProfile(
            openai_supports_strict_tool_definition=settings.agent_strict_tools,
        ),
    )
    prompt = (
        settings.agent_system_prompt
        if settings.agent_system_prompt.strip()
        else SYSTEM_PROMPT.format(max_iterations=settings.agent_max_iterations)
    )
    agent = Agent(
        model,
        system_prompt=prompt,
        deps_type=AgentDeps,
        output_type=str,
    )

    @agent.tool(strict=False)
    async def retrieve(
        ctx: RunContext[AgentDeps],
        queries: list[str],
    ) -> list[RetrievalResult]:
        """Search the vector database with one or more queries.

        Returns documents, metadata, and relevance scores.
        """
        return await _run_retrieve(ctx.deps, queries)

    return agent


async def _retrieve_all(
    queries: list[str],
    collection_names: list[str],
    fetch_k: int,
    rerank_k: int | None,
) -> list[RetrievalResult]:
    """Embed ``queries`` and run one retrieval per query, in order."""
    vectors, sparse_vectors, use_native_hybrid = await embed_dense_and_sparse(queries)
    results: list[RetrievalResult] = []
    for query_text, query_vector, sparse_vec in zip(queries, vectors, sparse_vectors, strict=True):
        texts, metadatas, distances = await retrieve_one_query(
            query_text,
            query_vector,
            sparse_vec,
            collection_names,
            fetch_k,
            use_native_hybrid,
            rerank_k=rerank_k,
        )
        log.debug("Retrieve for %r: %d documents found", query_text, len(texts))
        results.append(RetrievalResult(texts=texts, metadatas=metadatas, distances=distances))
    return results


async def _run_retrieve(deps: AgentDeps, queries: list[str]) -> list[RetrievalResult]:
    """Body of the ``retrieve`` tool, module-level so it's unit-testable."""
    log.info("Agent tool 'retrieve' called with %d queries", len(queries))
    log.debug("Agent tool 'retrieve' queries: %s", sanitize_for_log(queries))

    if not queries:
        # The system prompt tells the model to call retrieve([]) when no
        # retrieval is needed (greetings, small talk). Mark full_results so
        # the no-tool-call fallback doesn't fire, and skip the embedding
        # round-trip — OpenAI-compatible APIs reject an empty input list.
        deps.full_results = deps.full_results or []
        return []

    all_results = await _retrieve_all(queries, deps.collection_names, deps.fetch_k, deps.k)

    # Record this round's queries, per-query hit counts and top scores so
    # the agent's search/retry behaviour is observable after the run.
    deps.round_stats.append(
        {
            "queries": list(queries),
            "hit_counts": [len(r.texts) for r in all_results],
            "top_scores": [[round(d, 4) for d in r.distances[:3]] for r in all_results],
        }
    )

    # Accumulate full results across retries (dedup happens downstream)
    deps.full_results = (deps.full_results or []) + all_results

    return _build_previews(
        all_results,
        max_chars=settings.agent_tool_preview_chars,
        preview_k=settings.agent_preview_k,
    )


_agent: Agent[AgentDeps, str] | None = None


def _get_agent() -> Agent[AgentDeps, str]:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


async def close_client() -> None:
    """Close the agent's httpx transport. Called from the lifespan shutdown."""
    global _agent, _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None
    _agent = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _parse_fallback_queries(output: str) -> list[str] | None:
    """Try to extract queries from agent text output (when it skips tool calling).

    Takes the outermost ``{...}`` after the last ``[TOOL_CALLS]`` marker (or in
    the whole output when absent), which covers both plain JSON
    (``{"queries": [...]}``) and Mistral tool-call text
    (``[TOOL_CALLS]retrieve{"queries": [...]}``) even with braces in prose before it.
    """
    output = output.split("[TOOL_CALLS]")[-1]
    try:
        data = json.loads(output[output.index("{") : output.rindex("}") + 1])
    except (ValueError, TypeError):  # no braces, or invalid JSON
        return None
    if not isinstance(data, dict) or not isinstance(data.get("queries"), list):
        return None
    return [q for q in data["queries"] if isinstance(q, str) and q.strip()]


_PREVIEW_META_FIELDS: tuple[str, ...] = (
    "source",
    "title",
    "page",
    "headers",
    "languages",
    "collection_type",
)


def _preview_meta(meta: dict) -> dict:
    """Pick the subset of chunk metadata the retrieval-specialist agent sees.

    Structural fields (``page``, ``headers``) help the agent grade relevance —
    e.g. a chunk under ``headers=["Privacy", "Foundational Principles"]`` is
    a strong topical signal even when the body text is vague. ``source`` and
    ``collection_type`` ground the chunk's provenance. ``title`` (the
    document's own title from extractor metadata) and ``languages`` (detected
    language codes) are extra topical and language-match signals.

    Empty values are dropped so the LLM doesn't burn context on
    ``"page": null``; ``source`` is always present (``""`` if unset). The full
    meta still flows through ``deps.full_results`` to the final response.
    """
    out = {f: meta[f] for f in _PREVIEW_META_FIELDS if meta.get(f) not in (None, "", [], {})}
    out.setdefault("source", "")
    return out


def _build_previews(
    all_results: list[RetrievalResult],
    *,
    max_chars: int,
    preview_k: int,
) -> list[RetrievalResult]:
    """Build the deduped, truncated, capped preview list returned to the LLM.

    Full results stay in AgentDeps.full_results (used by the final response);
    previews are bounded by preview_k to keep the agent's context window
    under control across iterations. Each preview carries ``source`` plus
    any structural fields (``page``, ``headers``, ``collection_type``) the
    chunk has — see ``_preview_meta``. Results are interleaved across queries
    (see :func:`interleave_dedup`) so the grader sees each query's best hits,
    matching what :func:`dedup_topk` returns in the final payload.
    """
    rows = list(islice(interleave_dedup(all_results), preview_k))
    return [
        RetrievalResult(
            texts=[t[:max_chars] + "..." if len(t) > max_chars else t for t, _, _ in rows],
            metadatas=[_preview_meta(m) for _, m, _ in rows],
            distances=[d for _, _, d in rows],
        )
    ]


def _respond(results: list[RetrievalResult], k: int, label: str = "") -> SearchResponse:
    """Final payload: interleaved, deduped top-``k`` as a single result set."""
    texts, metadatas, distances = dedup_topk(results, k)
    log.info("Returning %d deduplicated results%s", len(texts), label)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Final payload%s: %s", label, _source_score_pairs(metadatas, distances))
    return SearchResponse(documents=[texts], metadatas=[metadatas], distances=[distances])


def _source_score_pairs(metadatas: list[dict], distances: list[float]) -> list[tuple[str, float]]:
    """``[(source, rounded_score), …]`` for DEBUG logging of a result set."""
    return [
        (meta.get("source", ""), round(dist, 4))
        for meta, dist in zip(metadatas, distances, strict=True)
    ]


async def _retrieve_raw_queries(
    queries: list[str],
    collection_names: list[str],
    fetch_k: int,
    k: int,
) -> list[RetrievalResult]:
    """Retrieve the user's *original* queries, before the agent rewrites them.

    Seeded into ``AgentDeps.full_results`` ahead of the agent loop so the
    unmodified question is always searched and merged (via
    :func:`interleave_dedup`) with the agent's own retrievals. This keeps a
    relevant chunk reachable even when the agent's reformulations drift toward
    keyword/web-search phrasing that matches boilerplate instead of content.
    Failures are swallowed — this is a recall safety-net, not a hard dependency.
    """
    try:
        return await _retrieve_all(queries, collection_names, fetch_k, k)
    except Exception:
        log.exception("Raw-query seed retrieval failed; continuing with agent-only results")
        return []


async def agentic_search(request: SearchRequest) -> SearchResponse:
    """Run the PydanticAI agent loop for agentic retrieval.

    The agent generates 1-2 queries and calls the ``retrieve`` tool once. It
    accepts the results if **any** returned document is on-topic, and only
    retries with rewritten queries when results are completely off-topic.
    Bounded by ``AGENT_MAX_ITERATIONS`` and ``AGENT_TIMEOUT`` (wall-clock).

    Full retrieval results accumulate on ``AgentDeps.full_results`` across
    iterations; only truncated previews go back to the LLM. On timeout, or
    when the agent emits queries as text instead of calling the tool, the
    fallback path performs a direct vector search using
    :func:`_parse_fallback_queries` to recover the queries.
    """
    queries = request.queries
    if not queries and request.messages:
        queries = extract_queries_from_messages(request.messages)

    if not queries:
        return SearchResponse(documents=[], metadatas=[], distances=[])

    k = request.k
    # fetch_k is the agent's internal candidate pool, decoupled from the
    # user-facing k. The wide pool feeds RRF / grading; final response is
    # still trimmed to k by dedup_topk.
    fetch_k = settings.agent_fetch_k

    agent = _get_agent()
    deps = AgentDeps(
        collection_names=request.collection_names,
        k=k,
        fetch_k=fetch_k,
    )

    # Always search the user's original query, independent of how the agent
    # rewrites it; the agent's own retrievals append to this seed and the lot
    # is merged by interleave_dedup. Guards recall against query-rewrite drift.
    if settings.agent_include_raw_query:
        seeded = await _retrieve_raw_queries(queries, deps.collection_names, fetch_k, k)
        if seeded:
            deps.full_results = seeded
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Seeded %d raw-query result set(s) before agent loop: %s",
                    len(seeded),
                    [_source_score_pairs(r.metadatas, r.distances) for r in seeded],
                )

    if request.messages:
        recent = request.messages[-settings.agent_conversation_history_messages :]
        conversation = "\n".join(f"{m.role}: {m.content}" for m in recent)
        user_prompt = (
            f"Analyze the following conversation and find the most relevant "
            f"documents for the user's latest information need.\n\n"
            f"Today's date: {date.today().isoformat()}\n\n"
            f"Conversation:\n{conversation}\n"
        )
    else:
        user_prompt = f"Find the most relevant documents for: {'; '.join(queries)}\n"

    # If Open WebUI passed a custom query generation template, include it
    # so the agent reflects admin-configured guidelines.
    if request.retrieval_query_generation_prompt_template:
        user_prompt += (
            f"\nAdditional query generation guidelines:\n"
            f"{request.retrieval_query_generation_prompt_template}\n"
        )

    user_prompt += (
        f"\nSearch across collections: {', '.join(request.collection_names)}\n"
        f"Return up to {k} results."
    )

    log.info(
        "Agentic search: queries=%d, collections=%s",
        len(queries),
        request.collection_names,
    )
    log.debug("Agentic search queries: %s", sanitize_for_log(queries))

    try:
        async with metrics.time_stage("agent_loop"):
            result = await asyncio.wait_for(
                agent.run(user_prompt, deps=deps, model_settings={"temperature": 0}),
                timeout=settings.agent_timeout,
            )
    except TimeoutError:
        metrics.agent_timeouts_total.inc()
        log.warning("Agent timed out after %ds, returning partial results", settings.agent_timeout)
        return _respond(deps.full_results or [], k, " (timeout)")

    # Fallback: if agent didn't call retrieve, do direct search
    if deps.full_results is None:
        metrics.agent_fallback_total.inc()
        log.warning("Agent did not call retrieve tool — falling back to direct search.")
        log.debug(
            "Agent fallback output (truncated): %s",
            result.output[:500] if result.output else "(empty)",
        )
        try:
            fallback_queries = _parse_fallback_queries(result.output) or queries
            # rerank_k=None — fallback intentionally returns raw vector results.
            deps.full_results = await _retrieve_all(
                fallback_queries, request.collection_names, fetch_k, None
            )
        except Exception:
            log.exception("Agent fallback direct search failed")
            deps.full_results = []

    retrieval_results = deps.full_results or []
    run_usage = result.usage
    log.info(
        "Agentic search complete: %d result sets, usage=%s",
        len(retrieval_results) if retrieval_results else 0,
        run_usage,
    )

    # Walk the model responses to surface token usage by role per step (the
    # queries built each round are in ``deps.round_stats``). ``retrieve_rounds > 1``
    # means a corrective retry happened (the agent judged a round off-topic and
    # searched again) — the closest observable signal of the retry decision.
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    step = 0
    retrieve_rounds = 0
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            step += 1
            u = msg.usage
            tool_calls = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            role = "retrieve" if tool_calls else "evaluate"
            if tool_calls:
                retrieve_rounds += 1
            metrics.agent_tokens_total.labels(role=role).inc(u.input_tokens + u.output_tokens)
            log.debug(
                "Agent step %d/%d (%s): model=%s, input_tokens=%d, "
                "output_tokens=%d, total_tokens=%d",
                step,
                run_usage.requests,
                role,
                msg.model_name or settings.agent_model,
                u.input_tokens,
                u.output_tokens,
                u.input_tokens + u.output_tokens,
            )

    metrics.agent_iterations.observe(run_usage.requests)
    if retrieve_rounds > 1:
        metrics.agent_retries_total.inc()
        log.info("Agent performed %d retrieve rounds (retry occurred)", retrieve_rounds)
    if deps.round_stats:
        log.debug("Agent round stats: %s", deps.round_stats)

    return _respond(retrieval_results, k)
