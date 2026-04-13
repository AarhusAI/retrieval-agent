# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic retrieval service for Open WebUI, replacing the `retrieval-service` POC. A standalone FastAPI microservice that wraps RAG vector search against Qdrant in an LLM-driven reasoning loop — adding query rewriting, decomposition, and relevance grading with retry. Uses **PydanticAI** (`pydantic-ai-slim[openai]`) for the agent loop (tool-calling, structured output, iteration control), with retrieval services (embedding, Qdrant, BM25, reranking) staying as direct custom code. Agent LLM calls go through the existing LiteLLM proxy in the parent stack. Designed to sit alongside the main openwebui-docker deployment and be called via Open WebUI's `RAG_EXTERNAL_RETRIEVAL_API_KEY` mechanism.

## Build & Run

All commands use [Task](https://taskfile.dev/) and run inside the Docker container (service name: `retrieval`). Python 3.11+ required. The Dockerfile uses multi-stage builds: `dev` target includes test/lint tools (ruff, pytest), `prod` target has runtime deps only. Docker Compose defaults to the `dev` target.

```shell
# First-time setup
cp .env.example .env        # then edit API_KEY and connection settings
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

# Run a single test file or test class
docker compose exec retrieval pytest tests/services/test_agent.py -v
docker compose exec retrieval pytest tests/services/test_agent.py::TestAgenticSearch -v
docker compose exec retrieval pytest tests/services/test_agent.py::TestAgenticSearch::test_empty_queries -v

# CI (lint + test)
task ci

# Production image
task build:image             # build + push to ghcr.io/aarhusai/retrieval-agent
task build:image TAG=v1.0.0  # with specific tag
```

## Architecture

Single FastAPI app (`app/main.py`) with two endpoints:

- **`POST /search`** — Bearer-token auth via `API_KEY`. Accepts either `queries` (explicit search strings) or `messages` (chat history for query extraction), plus `collection_names` and `k`. Returns `{documents, metadatas, distances}`.
- **`GET /health`** — healthcheck.

### Dual-Input Pattern

`SearchRequest` (`app/models.py`) accepts two input modes reflecting how Open WebUI calls this service:
- **`queries`** — pre-formed search strings (Open WebUI's default mode)
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
A PydanticAI `Agent` with a `retrieve` tool wrapping the same retrieval services. The agent decides strategy (direct search, query rewriting, or decomposition), calls the tool, grades results, and retries with rewritten queries if poor (Corrective RAG, up to `AGENT_MAX_ITERATIONS` attempts). Tool results are stored in `AgentDeps.full_results` as a side-channel — only truncated previews (`agent_tool_preview_chars`) go to the LLM to save tokens. If the agent fails to call the retrieve tool, a fallback does direct vector search.

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

## Rules

- **Never read `.env` files.** They contain secrets (API keys, credentials). Use `.env.example` to understand available settings.
- **Always run Python commands inside the Docker container.** `pip install`, `pytest`, `ruff`, and any other project commands must be executed via `docker compose exec retrieval ...` (or `docker compose run` if the container is not running). Never install or run Python tooling on the host.

## Key Constraints

- The embedding model and query prefix **must** be identical to what Open WebUI uses for indexing, or vector search will return garbage.
- The Qdrant multitenancy mapping must mirror Open WebUI's `qdrant_multitenancy.py`. If upstream changes that mapping, this service breaks.
- BM25 hybrid search scrolls entire collections from Qdrant on every query — only viable for small collections.
- Agentic mode adds 2–5x latency and 2–4x token cost per query vs the linear pipeline. Use a small, fast model (e.g. GPT-4o-mini) for agent decisions.
- `AGENT_MAX_ITERATIONS` bounds the corrective RAG loop to prevent runaway retries.
