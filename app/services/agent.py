"""
PydanticAI agent for agentic RAG — query analysis, rewriting, decomposition,
retrieval, and corrective relevance grading with retry.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
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
relevant documents for a user's query.

You have access to tools for searching a vector database. For each query:

1. ANALYZE the query — decide if it can be searched directly, needs rewriting \
   for clarity, or should be decomposed into sub-queries. If conversation context \
   is provided, use it to resolve vague references (e.g. "this", "that", "more about it") \
   and rewrite the query to be self-contained.
2. SEARCH using the retrieve tool.
3. GRADE the results — if they are not relevant to the original query, rewrite \
   the query and try again (up to {max_iterations} attempts).
4. Return the final results.

Keep queries concise and focused. When rewriting, make the query more specific \
and unambiguous. When decomposing, break into 2-3 independent sub-queries.\
"""


def _build_agent() -> Agent[AgentDeps, str]:
    """Build the PydanticAI agent. Called once at module level."""
    model = OpenAIChatModel(
        settings.agent_model,
        provider=OpenAIProvider(
            base_url=settings.agent_api_base_url or None,
            api_key=settings.agent_api_key or None,
        ),
    )
    agent = Agent(
        model,
        system_prompt=SYSTEM_PROMPT.format(max_iterations=settings.agent_max_iterations),
        deps_type=AgentDeps,
        output_type=str,
    )

    @agent.tool
    async def retrieve(
        ctx: RunContext[AgentDeps],
        queries: list[str],
    ) -> list[RetrievalResult]:
        """Search the vector database with one or more queries. Returns documents, metadata, and relevance scores."""
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

        # Return truncated previews to the LLM to save tokens
        max_chars = settings.agent_tool_preview_chars
        previews = []
        for r in all_results:
            previews.append(
                RetrievalResult(
                    texts=[t[:max_chars] + "..." if len(t) > max_chars else t for t in r.texts],
                    metadatas=r.metadatas,
                    distances=r.distances,
                )
            )
        return previews

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


def _dedup_results(
    results: list[RetrievalResult], k: int
) -> tuple[list[str], list[dict], list[float]]:
    """Deduplicate and limit results across all retrieval passes."""
    seen: set[str] = set()
    texts: list[str] = []
    metadatas: list[dict] = []
    distances: list[float] = []

    for result in results:
        for text, meta, dist in zip(
            result.texts, result.metadatas, result.distances, strict=True
        ):
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

    user_prompt = f"Find the most relevant documents for: {'; '.join(queries)}\n"

    if request.messages:
        conversation = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
        user_prompt += f"\nConversation context:\n{conversation}\n"

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

    # Use full results from deps (side-channel), not the agent's text output
    retrieval_results = deps.full_results or []
    log.info(
        "Agentic search complete: %d result sets, usage=%s",
        len(retrieval_results) if retrieval_results else 0,
        result.usage(),
    )
    texts, metadatas, distances = _dedup_results(retrieval_results, k)
    log.info("Returning %d deduplicated results", len(texts))

    return SearchResponse(
        documents=[texts],
        metadatas=[metadatas],
        distances=[distances],
    )
