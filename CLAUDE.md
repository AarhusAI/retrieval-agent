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

**Linear pipeline** (`linear_search`): query resolution → embed → vector search across collections concurrently → optional BM25 RRF fusion → optional cross-encoder rerank → dedup by MD5 → top-k. See `README.md` for the diagram.

**Agentic pipeline** (`app/services/agent.py`): a PydanticAI `Agent` with a `retrieve` tool wrapping the same primitives. The agent generates 1-2 queries, calls the tool, accepts results if any document is on-topic, and only retries with rewritten queries when results are completely off-topic (bounded by `AGENT_MAX_ITERATIONS`).

Three pieces of agent behaviour worth knowing because they are not obvious from any single file:

- **Side-channel results.** Full retrieval results (all metadata) are stashed on `AgentDeps.full_results`; the `retrieve` tool returns only previews to the LLM — text truncated to `AGENT_TOOL_PREVIEW_CHARS`, metadata reduced to `source`, dedup by MD5 across queries. The final pipeline output comes from `full_results`, not from anything the LLM sees. This keeps token usage low on small-context models.
- **Fallback parser.** If the agent emits queries as text instead of calling the tool, `_parse_fallback_queries()` (`app/services/agent.py`) extracts them — handles plain JSON and Mistral `[TOOL_CALLS]` syntax. Wrapped in exception handling: embedding/Qdrant failures inside the fallback return empty results, not 500s.
- **Timeout & accumulation.** `AGENT_TIMEOUT` is wall-clock. On timeout, whatever the `retrieve` tool already wrote to `AgentDeps.full_results` is returned. Multiple `retrieve` calls (corrective retries) **append** to `full_results`; they don't overwrite.

When reranking is enabled, initial retrieval fetches `k * INITIAL_RETRIEVAL_MULTIPLIER` candidates. All GPU/LLM services (embedding, reranking, agent) are external OpenAI-compatible APIs — no local models.

### Qdrant Multitenancy Mapping

`_get_collection_and_tenant_id()` in `app/services/qdrant.py` mirrors Open WebUI's `qdrant_multitenancy.py` exactly: collection names like `user-memory-*`, `file-*`, `web-search-*`, hex-hash, and knowledge collections are routed to shared Qdrant collections with tenant filters. **If upstream changes that mapping, this service breaks silently** — vector search will hit the wrong collection or filter on the wrong tenant. Re-check this function any time the Open WebUI fork is bumped.

### Testing

Tests use `pytest-asyncio` with `asyncio_mode = "auto"`. `tests/conftest.py` sets env vars **before any `app` imports** so tests don't accidentally hit real services — that's why test imports may look ordered oddly. External dependencies (Qdrant, embedding API, PydanticAI agent) are mocked.

## Configuration

All config via environment variables, loaded by pydantic-settings in `app/config.py`. See `.env.example` and `README.md` for the full list and defaults. The settings that matter beyond their docstrings — because they are **contracts with other systems**, not knobs:

- `API_KEY` must equal Open WebUI's `RAG_EXTERNAL_RETRIEVAL_API_KEY`.
- `EMBEDDING_MODEL` and especially `EMBEDDING_QUERY_PREFIX` must match Open WebUI's RAG config exactly. The prefix is applied to query strings in `app/services/embedding.py` before embedding; if it doesn't match what indexing used, vector search returns garbage. Note: query-generation output is not re-prefixed — the embedding stage adds the prefix, not the LLM.
- `QDRANT_MULTITENANCY` must match Open WebUI's setting (see Qdrant Multitenancy Mapping above).
- `AGENT_*` (`AGENT_MODEL`, `AGENT_API_BASE_URL`, `AGENT_API_KEY`, etc.) drives both the agentic loop and the linear pipeline's query generation. Defaults to the LiteLLM proxy at `http://litellm:4000/v1`. Use a small/fast model.
- `RETRIEVAL_QUERY_GENERATION_PROMPT_TEMPLATE` — optional override for the query-generation prompt (built-in prompt is used when empty).
- `AGENT_SYSTEM_PROMPT` — same idea for the agent loop.
- `AGENT_STRICT_TOOLS` — disable for models that don't support PydanticAI strict tool definitions (combine with the fallback parser).
- `DEBUG=true` enables debug logging on the `app` logger, which includes per-step token usage for agent LLM calls and query generation.

## Rules

- **Never read `.env` files.** They contain secrets (API keys, credentials). Use `.env.example` to understand available settings.
- **Always run Python commands inside the Docker container.** `pip install`, `pytest`, `ruff`, and any other project commands must be executed via `docker compose exec retrieval ...` from the repo root (or via the `task` wrapper). Never install or run Python tooling on the host.

## Failure-mode notes

- **BM25 scrolls entire collections** from Qdrant — only viable for small ones. The result is cached in-memory for `BM25_CACHE_TTL_SECONDS` (default 300s) to avoid re-scrolling on every query.
- **Reranker fails open**: HTTP/connection errors fall back to unranked results — a missing reranker won't take down search. Treat this as deliberate.
- **Qdrant errors are loud on purpose**: `vector_search` catches only `UnexpectedResponse` (e.g. collection not found). Connection errors and auth failures propagate as 500s so they're visible.
- The agentic loop is bounded by both `AGENT_MAX_ITERATIONS` (per-loop iterations) and `AGENT_TIMEOUT` (hard wall-clock). Tool results accumulate across iterations in the conversation history — watch the agent model's context window.
