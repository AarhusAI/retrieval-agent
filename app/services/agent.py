"""
PydanticAI agent for agentic RAG — query analysis, rewriting, decomposition,
retrieval, and corrective relevance grading with retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings
from app.models import SearchRequest, SearchResponse
from app.services.pipeline import (
    embed_dense_and_sparse,
    extract_queries_from_messages,
    retrieve_one_query,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """Result from a single retrieval pass."""

    texts: list[str]
    metadatas: list[dict]
    distances: list[float]


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

When grading relevance, use any structural metadata each result exposes — \
``headers`` (the section/heading breadcrumb a chunk sits under) and ``page`` \
(page number in the source document) are strong topical signals even when \
the chunk's body text is terse or generic.

Keep queries concise and focused. Prefer a single well-crafted query over \
multiple overlapping ones.\
"""


def _build_agent() -> Agent[AgentDeps, str]:
    """Build the PydanticAI agent. Called once at module level."""
    model = OpenAIChatModel(
        settings.agent_model,
        provider=OpenAIProvider(
            base_url=settings.agent_api_base_url or None,
            api_key=settings.agent_api_key or None,
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
        log.info("Agent tool 'retrieve' called with queries=%s", queries)
        vectors, sparse_vectors, use_native_hybrid = await embed_dense_and_sparse(queries)
        all_results: list[RetrievalResult] = []

        for query_text, query_vector, sparse_vec in zip(
            queries, vectors, sparse_vectors, strict=True
        ):
            texts, metadatas, distances = await retrieve_one_query(
                query_text,
                query_vector,
                sparse_vec,
                ctx.deps.collection_names,
                ctx.deps.fetch_k,
                use_native_hybrid,
                rerank_k=ctx.deps.k,
            )
            log.info("Retrieve for %r: %d documents found", query_text, len(texts))
            all_results.append(
                RetrievalResult(texts=texts, metadatas=metadatas, distances=distances)
            )

        # Accumulate full results across retries (dedup happens downstream)
        ctx.deps.full_results = (ctx.deps.full_results or []) + all_results

        return _build_previews(
            all_results,
            max_chars=settings.agent_tool_preview_chars,
            preview_k=settings.agent_preview_k,
        )

    return agent


_agent: Agent[AgentDeps, str] | None = None


def _get_agent() -> Agent[AgentDeps, str]:
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _parse_fallback_queries(output: str) -> list[str] | None:
    """Try to extract queries from agent text output (when it skips tool calling).

    Handles two formats:
    - Plain JSON: {"queries": ["q1", "q2"]}
    - Mistral tool-call text: [TOOL_CALLS]retrieve{"queries": ["q1", "q2"]}
    """
    # Try plain JSON first
    try:
        data = json.loads(output)
        if isinstance(data, dict) and "queries" in data:
            return [q for q in data["queries"] if isinstance(q, str) and q.strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    # Handle Mistral-style [TOOL_CALLS]function_name{...} format
    match = re.search(r"\[TOOL_CALLS\]\w+(\{.+)", output, re.DOTALL)
    if match:
        try:
            raw = match.group(1)
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start != -1 and end > 0:
                data = json.loads(raw[start:end])
                if isinstance(data, dict) and "queries" in data:
                    return [q for q in data["queries"] if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, TypeError):
            pass

    return None


_PREVIEW_META_FIELDS: tuple[str, ...] = ("source", "page", "headers", "collection_type")


def _preview_meta(meta: dict) -> dict:
    """Pick the subset of chunk metadata the retrieval-specialist agent sees.

    Structural fields (``page``, ``headers``) help the agent grade relevance —
    e.g. a chunk under ``headers=["Privacy", "Foundational Principles"]`` is
    a strong topical signal even when the body text is vague. ``source`` and
    ``collection_type`` ground the chunk's provenance.

    Missing / empty / falsy values are dropped so the LLM doesn't burn context
    on ``"page": null`` or ``"headers": []``. The full meta still flows
    through ``deps.full_results`` to the final ``SearchResponse``; this is
    purely about what the agent sees mid-loop.
    """
    out: dict = {}
    for key in _PREVIEW_META_FIELDS:
        value = meta.get(key)
        if value in (None, "", [], {}):
            continue
        out[key] = value
    # ``source`` is the only field every preview should carry — keep an empty
    # string if it wasn't set, matching the previous contract.
    if "source" not in out:
        out["source"] = ""
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
    chunk has — see ``_preview_meta``.
    """
    seen: set[str] = set()
    preview_texts: list[str] = []
    preview_metas: list[dict] = []
    preview_distances: list[float] = []

    for r in all_results:
        if len(preview_texts) >= preview_k:
            break
        for text, meta, dist in zip(r.texts, r.metadatas, r.distances, strict=True):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in seen:
                continue
            seen.add(text_hash)
            preview_texts.append(text[:max_chars] + "..." if len(text) > max_chars else text)
            preview_metas.append(_preview_meta(meta))
            preview_distances.append(dist)
            if len(preview_texts) >= preview_k:
                break

    return [
        RetrievalResult(
            texts=preview_texts,
            metadatas=preview_metas,
            distances=preview_distances,
        )
    ]


def _dedup_results(
    results: list[RetrievalResult], k: int
) -> tuple[list[str], list[dict], list[float]]:
    """Deduplicate and limit results across all retrieval passes."""
    seen: set[str] = set()
    texts: list[str] = []
    metadatas: list[dict] = []
    distances: list[float] = []

    for result in results:
        for text, meta, dist in zip(result.texts, result.metadatas, result.distances, strict=True):
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in seen:
                continue
            seen.add(text_hash)
            texts.append(text)
            metadatas.append(meta)
            distances.append(dist)
            if len(texts) >= k:
                return texts, metadatas, distances

    return texts, metadatas, distances


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
    # still trimmed to k by _dedup_results.
    fetch_k = settings.agent_fetch_k

    agent = _get_agent()
    deps = AgentDeps(
        collection_names=request.collection_names,
        k=k,
        fetch_k=fetch_k,
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

    log.info("Agentic search: queries=%s, collections=%s", queries, request.collection_names)

    try:
        result = await asyncio.wait_for(
            agent.run(user_prompt, deps=deps, model_settings={"temperature": 0}),
            timeout=settings.agent_timeout,
        )
    except TimeoutError:
        log.warning("Agent timed out after %ds, returning partial results", settings.agent_timeout)
        retrieval_results = deps.full_results or []
        texts, metadatas, distances = _dedup_results(retrieval_results, k)
        return SearchResponse(
            documents=[texts],
            metadatas=[metadatas],
            distances=[distances],
        )

    # Fallback: if agent didn't call retrieve, do direct search
    if deps.full_results is None:
        log.warning(
            "Agent did not call retrieve tool — falling back to direct search. Agent output: %s",
            result.output[:500] if result.output else "(empty)",
        )
        try:
            fallback_queries = _parse_fallback_queries(result.output) or queries
            vectors, sparse_vectors, use_native_hybrid = await embed_dense_and_sparse(
                fallback_queries
            )
            fallback_results: list[RetrievalResult] = []
            for query_text, query_vector, sparse_vec in zip(
                fallback_queries, vectors, sparse_vectors, strict=True
            ):
                # rerank_k=None — fallback intentionally returns raw vector results.
                texts, metadatas, distances = await retrieve_one_query(
                    query_text,
                    query_vector,
                    sparse_vec,
                    request.collection_names,
                    fetch_k,
                    use_native_hybrid,
                    rerank_k=None,
                )
                fallback_results.append(
                    RetrievalResult(texts=texts, metadatas=metadatas, distances=distances)
                )
            deps.full_results = fallback_results
        except Exception:
            log.exception("Agent fallback direct search failed")
            deps.full_results = []

    retrieval_results = deps.full_results or []
    run_usage = result.usage()
    log.info(
        "Agentic search complete: %d result sets, usage=%s",
        len(retrieval_results) if retrieval_results else 0,
        run_usage,
    )

    # Log per-request token usage from each model response
    from pydantic_ai.messages import ModelResponse, ToolCallPart

    step = 0
    for msg in result.all_messages():
        if isinstance(msg, ModelResponse):
            step += 1
            u = msg.usage
            has_tool_calls = any(isinstance(p, ToolCallPart) for p in msg.parts)
            role = "retrieve" if has_tool_calls else "evaluate"
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
    texts, metadatas, distances = _dedup_results(retrieval_results, k)
    log.info("Returning %d deduplicated results", len(texts))

    return SearchResponse(
        documents=[texts],
        metadatas=[metadatas],
        distances=[distances],
    )
