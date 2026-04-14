# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic retrieval service for Open WebUI, replacing the `retrieval-service` POC. A standalone FastAPI microservice that wraps RAG vector search against Qdrant in an LLM-driven reasoning loop — adding query rewriting, decomposition, and relevance grading with retry. Uses **PydanticAI** (`pydantic-ai-slim[openai]`) for the agent loop (tool-calling, structured output, iteration control), with retrieval services (embedding, Qdrant, BM25, reranking) staying as direct custom code. Agent LLM calls go through the existing LiteLLM proxy in the parent stack. Designed to sit alongside the main openwebui-docker deployment and be called via Open WebUI's `RAG_EXTERNAL_RETRIEVAL_API_KEY` mechanism.

## Build & Run

All commands use [Task](https://taskfile.dev/) and run inside the Docker container (service name: `retrieval`). Python 3.11+ required. The Dockerfile uses multi-stage builds: `dev` target includes test/lint tools (ruff, pytest), `prod` target has runtime deps only. Docker Compose defaults to the `dev` target.

**Important:** The retrieval service is defined in a compose override file. All Docker commands must be run from the parent `openwebui-docker/` directory using both compose files:

```shell
cd /path/to/openwebui-docker
docker compose -f docker-compose.yml -f docker-compose.agentic-retrieval.yml exec retrieval <command>
```

```shell
# First-time setup
cp agentic-retrieval/.env.example agentic-retrieval/.env  # then edit API_KEY and connection settings
task setup                   # starts containers + installs dev deps

# Container management
task up                      # start containers (requires Traefik 'frontend' network)
task down                    # stop containers
task logs                    # tail retrieval container logs
task shell                   # open bash shell in the retrieval container

# Install deps (inside container)
task install                 # pip install '.[dev]'

# Lint & format
task lint                    # run all linters (ruff check + format --check)
task lint:fix                # auto-fix lint issues
task lint:format             # auto-format code

# Tests
task test                    # run all tests (pytest -v)
task test:coverage           # run tests with coverage report

# Run a single test file or test class (from parent dir)
docker compose -f docker-compose.yml -f docker-compose.agentic-retrieval.yml exec retrieval pytest tests/services/test_agent.py -v
docker compose -f docker-compose.yml -f docker-compose.agentic-retrieval.yml exec retrieval pytest tests/services/test_agent.py::TestAgenticSearch -v

# CI (lint + test)
task ci

# Production image
task build:image             # build + push to ghcr.io/aarhusai/retrieval-agent
task build:image TAG=v1.0.0  # with specific tag
```

## Architecture

Single FastAPI app (`app/main.py`) with two endpoints:

- **`POST /search`** — Bearer-token auth via `API_KEY`. Accepts either `queries` (explicit search strings) or `messages` (chat history for query extraction), plus `collection_names` and `k`. Returns `{documents, metadatas, distances}`.
- **`GET /health`** — liveness probe (always 200 if the process is running).
- **`GET /health/ready`** — readiness probe (verifies Qdrant connectivity, returns 503 if unreachable).

### Dual-Input Pattern

`SearchRequest` (`app/models.py`) accepts two input modes reflecting how Open WebUI calls this service:
- **`queries`** — pre-formed search strings (Open WebUI's default mode). Open WebUI runs its own query generation and sends the result.
- **`messages`** — full chat history; the service extracts/generates queries itself. When `ENABLE_QUERY_GENERATION=true` (linear pipeline) or in agentic mode, an LLM generates optimized queries from the conversation. Otherwise falls back to the last user message.

### Pipeline Router (`app/services/pipeline.py`)

`search()` routes to either pipeline based on `ENABLE_AGENTIC_RAG`:

**Linear pipeline** (`linear_search`):
1. Resolve queries — LLM generation from messages (`app/services/query_generation.py`) → explicit queries → last user message fallback
2. Embed queries via OpenAI-compatible API (`app/services/embedding.py`)
3. Vector search across collections concurrently (`app/services/qdrant.py`)
4. Optional BM25 hybrid fusion via Reciprocal Rank Fusion (`app/services/bm25.py`)
5. Optional cross-encoder reranking via `/v1/rerank` API (`app/services/reranker.py`)
6. Dedup by MD5 text hash, limit to k

**Agentic pipeline** (`app/services/agent.py`):
A PydanticAI `Agent` with a `retrieve` tool wrapping the same retrieval services. The agent generates 1-2 search queries, calls the tool, and accepts results if any document is on-topic. It only retries with rewritten queries if results are completely off-topic (up to `AGENT_MAX_ITERATIONS` attempts).

Tool results use a side-channel pattern: full results (with all metadata) are stored in `AgentDeps.full_results`, while only minimal previews go to the LLM — text truncated to `AGENT_TOOL_PREVIEW_CHARS`, metadata stripped to just `source`, and duplicates removed across queries by MD5 hash. This keeps token usage low for small-context models.

If the agent fails to call the retrieve tool, a fallback parses queries from the output (supports plain JSON and Mistral `[TOOL_CALLS]` format) and does direct vector search. The fallback is wrapped in exception handling — if embedding or Qdrant fails, it returns empty results rather than crashing.

The agent run is bounded by `AGENT_TIMEOUT` (default 60s wall-clock). On timeout, whatever partial results the retrieve tool has collected so far are returned. Multiple retrieve calls (corrective RAG retries) accumulate results rather than overwriting.

When reranking is enabled, initial retrieval fetches `k * INITIAL_RETRIEVAL_MULTIPLIER` candidates. All GPU/LLM services (embedding, reranking, agent) are configured as external OpenAI-compatible APIs — no local models.

### Qdrant Multitenancy Mapping (`app/services/qdrant.py`)

The service replicates Open WebUI's `qdrant_multitenancy.py` collection-routing logic: collection names like `user-memory-*`, `file-*`, `web-search-*`, hex-hash, and knowledge collections are mapped to shared Qdrant collections with tenant filters. This mapping **must stay in sync** with the Open WebUI fork.

### Testing

Tests use `pytest-asyncio` with `asyncio_mode = "auto"` (all async tests run automatically). `conftest.py` overrides env vars before any app imports to avoid hitting real services. Tests mock external dependencies (Qdrant, embedding API, PydanticAI agent) — no real services needed.

## Configuration

All config via environment variables (or `.env` file), loaded by pydantic-settings in `app/config.py`. See `.env.example` for all available settings and defaults. Key settings:

- `API_KEY` — must match `RAG_EXTERNAL_RETRIEVAL_API_KEY` in Open WebUI
- `EMBEDDING_MODEL` / `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY` — must match Open WebUI's RAG embedding config exactly
- `QDRANT_MULTITENANCY` — must match Open WebUI's Qdrant multitenancy setting
- `ENABLE_HYBRID_SEARCH` / `ENABLE_RERANKING` — toggle optional pipeline stages
- `ENABLE_QUERY_GENERATION` — toggle LLM-based query generation in the linear pipeline
- `ENABLE_AGENTIC_RAG` — toggle agentic mode (when `false`, falls back to linear pipeline)
- `AGENT_MODEL` / `AGENT_API_BASE_URL` / `AGENT_API_KEY` — LLM for both the PydanticAI agent loop and query generation (defaults to LiteLLM proxy at `http://litellm:4000/v1`; use a fast/cheap model like GPT-4o-mini)
- `AGENT_TOOL_PREVIEW_CHARS` — max characters per document preview sent to the agent LLM (default 200)
- `AGENT_STRICT_TOOLS` — enable strict tool definition validation in PydanticAI (default true; disable for models that don't support it)
- `AGENT_TIMEOUT` — wall-clock timeout in seconds for the agent run (default 60); returns partial results on timeout
- `AGENT_SYSTEM_PROMPT` — override the default agent system prompt (uses built-in prompt when empty)
- `BM25_CACHE_TTL_SECONDS` — TTL for in-memory BM25 index cache (default 300s); avoids re-scrolling Qdrant on every query
- `DEBUG` — set to `true` to enable debug logging for the `app` logger, which includes per-step token usage for agent LLM calls and query generation

## Rules

- **Never read `.env` files.** They contain secrets (API keys, credentials). Use `.env.example` to understand available settings.
- **Always run Python commands inside the Docker container.** `pip install`, `pytest`, `ruff`, and any other project commands must be executed via `docker compose -f docker-compose.yml -f docker-compose.agentic-retrieval.yml exec retrieval ...` from the parent `openwebui-docker/` directory. Never install or run Python tooling on the host.

## Key Constraints

- The embedding model and query prefix **must** be identical to what Open WebUI uses for indexing, or vector search will return garbage.
- The Qdrant multitenancy mapping must mirror Open WebUI's `qdrant_multitenancy.py`. If upstream changes that mapping, this service breaks.
- BM25 hybrid search scrolls entire collections from Qdrant — only viable for small collections. Results are cached in-memory with a TTL (`BM25_CACHE_TTL_SECONDS`, default 5 min) to avoid re-scrolling on every query.
- Agentic mode adds latency and token cost per query vs the linear pipeline. Use a small, fast model for agent decisions. Be aware of the agent model's context window — tool results accumulate in the conversation history.
- `AGENT_MAX_ITERATIONS` bounds the corrective RAG loop to prevent runaway retries. `AGENT_TIMEOUT` provides a hard wall-clock cap.
- The reranker gracefully falls back to unranked results on HTTP or connection errors — it won't take down the whole search if the reranker endpoint is unavailable.
- Qdrant `vector_search` only catches `UnexpectedResponse` (e.g. collection not found). Connection errors and auth failures propagate as 500s for proper error visibility.
