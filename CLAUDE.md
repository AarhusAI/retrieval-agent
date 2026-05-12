# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic retrieval service for Open WebUI, replacing the `retrieval-service` POC. A standalone FastAPI microservice that wraps RAG vector search against Qdrant in an LLM-driven reasoning loop — adding query rewriting, decomposition, and relevance grading with retry. Uses **PydanticAI** (`pydantic-ai-slim[openai]`) for the agent loop (tool-calling, structured output, iteration control), with retrieval services (embedding, Qdrant, BM25, reranking) staying as direct custom code. Agent LLM calls go through the existing LiteLLM proxy in the parent stack. Designed to sit alongside the main openwebui-docker deployment and be called via Open WebUI's `RAG_EXTERNAL_RETRIEVAL_API_KEY` mechanism.

## Build & Run

The repo is **standalone** — its own `docker-compose.yml`, its own `.env`, run from the repo root. (Earlier setups had it as an override on top of the parent `openwebui-docker/` stack; that's no longer the case.)

The Dockerfile is multi-stage: `dev` target has test/lint tools (ruff, pytest), `prod` target is runtime-only. Compose defaults to `dev`. Python 3.12 in the container; `pyproject.toml` requires `>=3.11`.

The `frontend` Docker network is **external** — created by Traefik in the parent stack, or manually via `docker network create frontend` for standalone use. `task up` will refuse to start without it.

All `task` commands proxy through `docker compose exec retrieval` (`Taskfile.yml:11-12`). See `README.md` for the full task catalogue. The non-obvious incantation worth keeping here — running a single test or test class:

```shell
docker compose exec retrieval pytest tests/services/test_agent.py::TestAgenticSearch -v
```

## Architecture

FastAPI app wired in `app/main.py` (lifespan, health probes, router include). Endpoints:

- **`POST /search`** — defined in `app/routes/search.py`. Bearer-token auth via `API_KEY` (`app/auth.py`). Accepts either `queries` (explicit search strings) or `messages` (chat history for query extraction), plus `collection_names` and `k`. Returns `{documents, metadatas, distances}`. The route is a thin shell — actual routing logic lives in `app/services/pipeline.py`.
- **`GET /health`** — liveness probe (always 200 if the process is running).
- **`GET /health/ready`** — readiness probe (verifies Qdrant connectivity, returns 503 if unreachable).

### Dual-Input Pattern

`SearchRequest` (`app/models.py`) accepts two input modes reflecting how Open WebUI calls this service:
- **`queries`** — pre-formed search strings (Open WebUI's default mode). Open WebUI runs its own query generation and sends the result.
- **`messages`** — full chat history; the service extracts/generates queries itself. When `ENABLE_QUERY_GENERATION=true` (linear pipeline) or in agentic mode, an LLM generates optimized queries from the conversation. Otherwise falls back to the last user message.

### Pipeline Router (`app/services/pipeline.py`)

`search()` routes to one of two pipelines based on `ENABLE_AGENTIC_RAG`. Both share the same retrieval primitives (embedding, Qdrant, BM25, reranking) — the difference is whether an LLM drives the loop.

**Linear pipeline** (`linear_search`): query resolution → embed (dense + optional sparse) → single Qdrant query per dense query, scoped to all `collection_names` via a `meta.collection_name IN (...)` filter → optional hybrid fusion (see below) → optional cross-encoder rerank → dedup by MD5 → top-k. See `README.md` for the diagram. The pre-Phase-3 per-collection `asyncio.gather` is gone — one logical query is one Qdrant call.

**Agentic pipeline** (`app/services/agent.py`): a PydanticAI `Agent` with a `retrieve` tool wrapping the same primitives. The agent generates 1-2 queries, calls the tool, accepts results if any document is on-topic, and only retries with rewritten queries when results are completely off-topic (bounded by `AGENT_MAX_ITERATIONS`).

Three pieces of agent behaviour worth knowing because they are not obvious from any single file:

- **Side-channel results.** Full retrieval results (all metadata) are stashed on `AgentDeps.full_results`; the `retrieve` tool returns only previews to the LLM — text truncated to `AGENT_TOOL_PREVIEW_CHARS`, metadata reduced to `source`, dedup by MD5 across queries. The final pipeline output comes from `full_results`, not from anything the LLM sees. This keeps token usage low on small-context models.
- **Fallback parser.** If the agent emits queries as text instead of calling the tool, `_parse_fallback_queries()` (`app/services/agent.py`) extracts them — handles plain JSON and Mistral `[TOOL_CALLS]` syntax. Wrapped in exception handling: embedding/Qdrant failures inside the fallback return empty results, not 500s.
- **Timeout & accumulation.** `AGENT_TIMEOUT` is wall-clock. On timeout, whatever the `retrieve` tool already wrote to `AgentDeps.full_results` is returned. Multiple `retrieve` calls (corrective retries) **append** to `full_results`; they don't overwrite.

When reranking is enabled, initial retrieval fetches `k * INITIAL_RETRIEVAL_MULTIPLIER` candidates. The agent path uses `AGENT_FETCH_K` instead — decoupled from `request.k` so a small Open WebUI `top_k` doesn't starve the grading pool; only `AGENT_PREVIEW_K` of those are previewed to the LLM. Dense embedding, reranker, and agent LLM are external OpenAI-compatible APIs; **sparse embedding runs in-process** via fastembed (see below).

### Qdrant Schema (Phase 3 — single collection)

All retrievable data lives in **one physical Qdrant collection** named by `QDRANT_INDEX` (default `ingestion_files`), populated by a separate external **ingestion service** that writes the Haystack-native `qdrant-haystack` schema:

- Chunk text at `payload.content`
- Metadata at `payload.meta.*`, with `meta.collection_name` carrying Open WebUI's logical collection identifier (e.g. `file-…`, `user-memory-…`, knowledge collection IDs)
- Dense vector under the named vector `text-dense`; sparse vector (when present) under `text-sparse`

`vector_search` in `app/services/qdrant.py` issues a single Qdrant query filtered by `meta.collection_name MatchAny (collection_names)`. The pre-Phase-3 multitenancy mapping (`_get_collection_and_tenant_id`, `qdrant_multitenancy.py` mirror, per-class shared collections, `tenant_id` filter, `QDRANT_COLLECTION_PREFIX`, `QDRANT_MULTITENANCY`) has been **removed entirely** — don't reintroduce that vocabulary. The contract is now with the ingestion service's schema, not with Open WebUI's `qdrant_multitenancy.py`.

### Hybrid Search (two paths)

`ENABLE_HYBRID_SEARCH=true` selects one of two fusion paths based on a runtime capability probe (`qdrant.has_sparse_vectors()`, cached for the process lifetime once a definite answer is obtained):

- **Native server-side** — when `text-sparse` exists on the configured collection. `app/services/sparse_embedding.py` runs `fastembed` in-process (default model `Qdrant/bm42-all-minilm-l6-v2-attentions`) to produce a sparse query vector; Qdrant's Query API does prefetch on both `text-dense` and `text-sparse` then RRF-fuses server-side. The sparse model is preloaded in the FastAPI lifespan when applicable so the first query doesn't pay the download/import cost. Set `SPARSE_QUERY_PROVIDER=none` to disable the sparse stage and force dense-only here.
- **Client-side BM25 fallback** — when the collection has no `text-sparse` vector. `app/services/bm25.py` scrolls the (filtered) Qdrant content into an in-memory BM25 index keyed on the **sorted tuple of `collection_names`** (different orderings share a cache entry), then `reciprocal_rank_fusion()` weights it with `HYBRID_BM25_WEIGHT`.

The two paths are mutually exclusive per call — the native path already RRF-fuses, so client-side fusion never runs on top of it (that would double-rank).

### Testing

Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. `tests/conftest.py` sets env vars **before any `app` imports** so tests don't accidentally hit real services — that's why test imports may look ordered oddly. External dependencies (Qdrant, embedding API, PydanticAI agent) are mocked.

## Configuration

All config via environment variables, loaded by pydantic-settings in `app/config.py`. See `.env.example` and `README.md` for the full list and defaults. The settings that matter beyond their docstrings — because they are **contracts with other systems**, not knobs:

- `API_KEY` must equal Open WebUI's `RAG_EXTERNAL_RETRIEVAL_API_KEY`.
- `EMBEDDING_MODEL` and `EMBEDDING_PREFIX_QUERY` must match what the **ingestion service** used at index time (for e5: `"query: "` on queries, `"passage: "` on documents; bge-m3 uses no prefix). The prefix is applied in `app/services/embedding.py` before embedding; mismatch → vector search returns garbage. Query-generation output is not re-prefixed — the embedding stage adds the prefix, not the LLM. (Note: this setting was renamed from the older `EMBEDDING_QUERY_PREFIX`.)
- `QDRANT_INDEX` is the single physical collection written by the ingestion service. There is no longer a `QDRANT_MULTITENANCY` / `QDRANT_COLLECTION_PREFIX` to coordinate with Open WebUI.
- `SPARSE_QUERY_MODEL` must match the sparse model the ingestion service used for indexing (no contract enforcement — just garbage out otherwise). `SPARSE_QUERY_PROVIDER=none` cleanly disables the sparse stage while keeping hybrid semantics intact.
- `AGENT_*` (`AGENT_MODEL`, `AGENT_API_BASE_URL`, `AGENT_API_KEY`, etc.) drives both the agentic loop and the linear pipeline's query generation. Defaults to the LiteLLM proxy at `http://litellm:4000/v1`. Use a small/fast model.
- `AGENT_FETCH_K` / `AGENT_PREVIEW_K` / `AGENT_CONVERSATION_HISTORY_MESSAGES` — agentic-recall knobs decoupled from `request.k`. Widen `FETCH_K` to feed the grader more candidates without inflating the LLM context; `PREVIEW_K` caps what the `retrieve` tool actually returns to the agent.
- `RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE` — optional override for the query-generation prompt (built-in prompt is used when empty).
- `AGENT_SYSTEM_PROMPT` — same idea for the agent loop.
- `AGENT_STRICT_TOOLS` — disable for models that don't support PydanticAI strict tool definitions (combine with the fallback parser).
- `DEBUG=true` enables debug logging on the `app` logger, which includes per-step token usage for agent LLM calls and query generation.

## Rules

- **Never read `.env` files.** They contain secrets (API keys, credentials). Use `.env.example` to understand available settings.
- **Always run Python commands inside the Docker container.** `pip install`, `pytest`, `ruff`, and any other project commands must be executed via `docker compose exec retrieval ...` from the repo root (or via the `task` wrapper). Never install or run Python tooling on the host.

## Failure-mode notes

- **BM25 scrolls entire collections** from Qdrant — only viable for small scopes. Only runs in the client-side fallback hybrid path (collection without `text-sparse`). Cached in-memory keyed on the sorted tuple of collection names for `BM25_CACHE_TTL_SECONDS` (default 300s).
- **Reranker fails open**: HTTP/connection errors fall back to unranked results — a missing reranker won't take down search. Treat this as deliberate.
- **Qdrant errors are loud on purpose**: `vector_search` catches only `UnexpectedResponse` (e.g. collection not found). Connection errors and auth failures propagate as 500s so they're visible.
- **Sparse capability is detected once.** `has_sparse_vectors()` caches the answer for the process lifetime. If the collection is recreated with a different vector config, call `reset_sparse_capability()` or restart the service — otherwise the wrong hybrid path keeps running. Cold-start before any ingest is the one case the cache deliberately skips (returns `False` without caching so first post-ingest query re-detects).
- The agentic loop is bounded by both `AGENT_MAX_ITERATIONS` (per-loop iterations) and `AGENT_TIMEOUT` (hard wall-clock). Tool results accumulate across iterations in the conversation history — watch the agent model's context window.

## `scripts/` — eval analysis (not in tests/)

Standalone CLIs for inspecting [promptfoo](https://promptfoo.dev/) eval JSON output and probing the reranker — not wired into `task` or pytest, run them directly:

- `scripts/eval_diff.py before.json after.json [--metric ...]` — diff two promptfoo runs by metric, grouped by source document. Default metric is `contentRetrieval`. Skips `difficulty: impossible` and `refusal: yes` tests unless flags say otherwise.
- `scripts/show_failures.py eval.json` — show failing tests with question, expected snippets, bot response, and retrieved sources. Same default skips as `eval_diff.py`.
- `scripts/rerank_check.py rank|truncate ...` — hit the configured `/v1/rerank` endpoint with a query + candidates (rank mode) or the same content at growing prefix lengths to probe tokenizer-level truncation (truncate mode). Reads `RERANKER_*` env vars; source `.env` first.
