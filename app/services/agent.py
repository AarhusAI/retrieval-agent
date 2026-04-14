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
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from app.config import settings
from app.models import ChatMessage, SearchRequest, SearchResponse
from app.services import embedding, qdrant
from app.services.bm25 import bm25_search, reciprocal_rank_fusion
from app.services.reranker import rerank

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output models
# ---------------------------------------------------------------------------


class QueryPlan(BaseModel):
    """The agent's decision on how to handle the incoming query."""

    strategy: Literal["direct", "rewrite", "decompose"]
    queries: list[str]


class RetrievalResult(BaseModel):
    """Result from a single retrieval pass."""

    texts: list[str]
    metadatas: list[dict]
    distances: list[float]


class GradedResult(BaseModel):
    """Agent's assessment of retrieval quality."""

    relevant: bool
    reason: str


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
    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT.format(max_iterations=settings.agent_max_iterations),
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
        vectors = await embedding.embed_queries(queries)
        all_results: list[RetrievalResult] = []

        for query_text, query_vector in zip(queries, vectors, strict=True):
            tasks = [
                qdrant.vector_search(coll, query_vector, ctx.deps.fetch_k)
                for coll in ctx.deps.collection_names
            ]
            results = await asyncio.gather(*tasks)

            merged_texts: list[str] = []
            merged_metadatas: list[dict] = []
            merged_distances: list[float] = []

            for result in results:
                merged_texts.extend(result.texts)
                merged_metadatas.extend(result.metadatas)
                merged_distances.extend(result.distances)

            # Optional hybrid search
            if settings.enable_hybrid_search and merged_texts:
                vector_ranked = list(zip(merged_texts, merged_distances, strict=True))
                bm25_tasks = [
                    bm25_search(coll, query_text, ctx.deps.fetch_k)
                    for coll in ctx.deps.collection_names
                ]
                bm25_results_per_coll = await asyncio.gather(*bm25_tasks)
                bm25_merged: list[tuple[str, float]] = []
                for bm25_res in bm25_results_per_coll:
                    bm25_merged.extend(bm25_res)

                fused = reciprocal_rank_fusion(
                    vector_ranked, bm25_merged, settings.hybrid_bm25_weight
                )
                text_to_meta = dict(zip(merged_texts, merged_metadatas, strict=True))
                merged_texts = [text for text, _ in fused]
                merged_distances = [score for _, score in fused]
                merged_metadatas = [text_to_meta.get(t, {}) for t in merged_texts]

            # Optional reranking
            if settings.enable_reranking and merged_texts:
                merged_texts, merged_metadatas, merged_distances = await rerank(
                    query_text, merged_texts, merged_metadatas, ctx.deps.k
                )

            log.info(
                "Retrieve for %r: %d documents found",
                query_text,
                len(merged_texts),
            )
            all_results.append(
                RetrievalResult(
                    texts=merged_texts,
                    metadatas=merged_metadatas,
                    distances=merged_distances,
                )
            )

        # Store full results in deps for later use in the response
        ctx.deps.full_results = all_results

        # Dedup across queries and build minimal previews for the LLM.
        # Full metadata is preserved in deps.full_results — previews only
        # include source name to save context window tokens.
        max_chars = settings.agent_tool_preview_chars
        seen: set[str] = set()
        preview_texts: list[str] = []
        preview_sources: list[str] = []
        preview_distances: list[float] = []

        for r in all_results:
            for text, meta, dist in zip(r.texts, r.metadatas, r.distances, strict=True):
                text_hash = hashlib.md5(text.encode()).hexdigest()
                if text_hash in seen:
                    continue
                seen.add(text_hash)
                preview_texts.append(text[:max_chars] + "..." if len(text) > max_chars else text)
                preview_sources.append(meta.get("source", ""))
                preview_distances.append(dist)

        return [
            RetrievalResult(
                texts=preview_texts,
                metadatas=[{"source": s} for s in preview_sources],
                distances=preview_distances,
            )
        ]

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


def extract_queries_from_messages(messages: list[ChatMessage]) -> list[str]:
    """Extract search queries from chat messages (last user message)."""
    for message in reversed(messages):
        if message.role == "user" and message.content.strip():
            return [message.content.strip()]
    return []


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
    match = re.search(r"\[TOOL_CALLS\]\w+(\{.*\})", output, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "queries" in data:
                return [q for q in data["queries"] if isinstance(q, str) and q.strip()]
        except (json.JSONDecodeError, TypeError):
            pass

    return None


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
    """
    Run the PydanticAI agent to perform agentic retrieval:
    1. Agent analyzes query and decides strategy (direct/rewrite/decompose)
    2. Agent calls retrieve tool (possibly multiple times with rewritten queries)
    3. Agent grades results and retries if needed
    4. Results are deduped and returned
    """
    queries = request.queries
    if not queries and request.messages:
        queries = extract_queries_from_messages(request.messages)

    if not queries:
        return SearchResponse(documents=[], metadatas=[], distances=[])

    k = request.k
    fetch_k = k
    if settings.enable_reranking:
        fetch_k = k * settings.initial_retrieval_multiplier

    agent = _get_agent()
    deps = AgentDeps(
        collection_names=request.collection_names,
        k=k,
        fetch_k=fetch_k,
    )

    if request.messages:
        conversation = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
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

    result = await agent.run(
        user_prompt,
        deps=deps,
        model_settings={"temperature": 0},
    )

    # Fallback: if agent didn't call retrieve, do direct search
    if deps.full_results is None:
        log.warning(
            "Agent did not call retrieve tool — falling back to direct search. Agent output: %s",
            result.output[:500] if result.output else "(empty)",
        )
        fallback_queries = _parse_fallback_queries(result.output) or queries
        vectors = await embedding.embed_queries(fallback_queries)
        fallback_results: list[RetrievalResult] = []
        for _query_text, query_vector in zip(fallback_queries, vectors, strict=True):
            tasks = [
                qdrant.vector_search(coll, query_vector, fetch_k)
                for coll in request.collection_names
            ]
            results = await asyncio.gather(*tasks)
            merged_texts: list[str] = []
            merged_metadatas: list[dict] = []
            merged_distances: list[float] = []
            for r in results:
                merged_texts.extend(r.texts)
                merged_metadatas.extend(r.metadatas)
                merged_distances.extend(r.distances)
            fallback_results.append(
                RetrievalResult(
                    texts=merged_texts,
                    metadatas=merged_metadatas,
                    distances=merged_distances,
                )
            )
        deps.full_results = fallback_results

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
