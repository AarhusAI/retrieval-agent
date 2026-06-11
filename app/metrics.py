"""Prometheus metrics for the retrieval-agent service.

Collectors are module-level singletons on the default registry. They are
always live — incrementing/observing is cheap and unconditional. The
``METRICS_ENABLED`` flag only gates whether the ``/metrics`` endpoint *exposes*
them, so call-sites stay guard-free and the numbers are correct the moment an
operator turns the endpoint on.

``prometheus_client`` collectors are thread-safe, so they're safe to touch from
``asyncio.to_thread`` workers (e.g. the BM25 scroll / Qdrant client calls).

Note — there is intentionally **no** ``instrument_stage`` helper like the
ingestion service has: that wraps a synchronous Haystack ``component.run``.
Every retrieval stage here is an ``async def`` coroutine, so we time stages
with the ``time_stage`` async context manager below instead.

Cardinality guard: never label a metric by ``collection_name`` or query text —
only the fixed enums declared here. ``code`` is a classified string
(``type(exc).__name__``), never a raw exception message.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from prometheus_client import Counter, Histogram

# Buckets tuned per metric family.
_REQUEST_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)
_STAGE_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)
_COUNT_BUCKETS = (0, 1, 2, 3, 5, 8, 10, 20, 50, 100)

# ---------------------------------------------------------------------------
# Request-level (owned by pipeline.search)
# ---------------------------------------------------------------------------

# Search attempts by pipeline and outcome. ``code`` is "none" on success, else
# the classified error (``type(exc).__name__``). Auth/validation 4xx rejects
# are NOT counted here — this tracks search-pipeline outcomes.
search_requests_total = Counter(
    "search_requests_total",
    "Search attempts by pipeline, outcome and classified error code.",
    ["pipeline", "outcome", "code"],
)

# Wall-clock of the whole /search call (query resolution → retrieval → response).
search_duration_seconds = Histogram(
    "search_duration_seconds",
    "Whole-request search wall-clock duration in seconds.",
    ["pipeline"],
    buckets=_REQUEST_BUCKETS,
)

# Documents returned to the caller (sum across all query result sets).
results_returned = Histogram(
    "results_returned",
    "Number of documents returned to the caller per search request.",
    buckets=_COUNT_BUCKETS,
)

# ---------------------------------------------------------------------------
# Stage-level
# ---------------------------------------------------------------------------

# Per-stage latency within a search. ``stage`` is one of:
# query_generation | embed_dense | embed_sparse | qdrant | bm25 | rerank | agent_loop.
retrieval_stage_duration_seconds = Histogram(
    "retrieval_stage_duration_seconds",
    "Per-stage wall-clock duration within a search, in seconds.",
    ["stage"],
    buckets=_STAGE_BUCKETS,
)

# Candidates pulled from Qdrant for a single query, before fusion/rerank/dedup.
candidates_fetched = Histogram(
    "candidates_fetched",
    "Candidate documents fetched from Qdrant per query (before fusion/rerank).",
    buckets=_COUNT_BUCKETS,
)

# Which fusion path actually ran: native server-side RRF (sparse present),
# client-side BM25 RRF fallback, or dense-only.
hybrid_path_total = Counter(
    "hybrid_path_total",
    "Retrieval fusion path taken per embed call.",
    ["path"],
)

# Reranker fail-open events (HTTP/connection error → unranked results).
reranker_failures_total = Counter(
    "reranker_failures_total",
    "Reranker requests that failed open and returned unranked results.",
)

# BM25 in-memory index cache effectiveness.
bm25_cache_total = Counter(
    "bm25_cache_total",
    "BM25 index cache lookups by result.",
    ["result"],
)

# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

# Model requests the agent made in a run (proxy for how much it looped).
agent_iterations = Histogram(
    "agent_iterations",
    "Model requests per agentic run (loop length).",
    buckets=_COUNT_BUCKETS,
)

# A corrective retry happened: the agent judged a round completely off-topic
# and issued another retrieve call.
agent_retries_total = Counter(
    "agent_retries_total",
    "Agentic runs in which a corrective retry (>1 retrieve round) occurred.",
)

# The agent run hit AGENT_TIMEOUT and returned partial accumulated results.
agent_timeouts_total = Counter(
    "agent_timeouts_total",
    "Agentic runs that timed out and returned partial results.",
)

# The model emitted queries as text instead of calling the retrieve tool.
agent_fallback_total = Counter(
    "agent_fallback_total",
    "Agentic runs that fell back to direct search (no tool call).",
)

# Token usage attributed to searching (tool-call steps) vs grading (eval steps).
agent_tokens_total = Counter(
    "agent_tokens_total",
    "Agent LLM tokens by step role.",
    ["role"],
)


@asynccontextmanager
async def time_stage(stage: str):
    """Time an async retrieval stage into ``retrieval_stage_duration_seconds``.

    Observes in ``finally`` so a failing stage still contributes its latency,
    and never swallows the stage's own exception.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        retrieval_stage_duration_seconds.labels(stage=stage).observe(time.perf_counter() - start)
