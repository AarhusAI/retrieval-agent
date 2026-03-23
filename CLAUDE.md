# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic retrieval service for Open WebUI, replacing the `retrieval-service` POC. A standalone FastAPI microservice that wraps RAG vector search against Qdrant in an LLM-driven reasoning loop — adding query rewriting, decomposition, and relevance grading with retry. Uses **PydanticAI** (`pydantic-ai-slim[openai]`) for the agent loop (tool-calling, structured output, iteration control), with retrieval services (embedding, Qdrant, BM25, reranking) staying as direct custom code. Agent LLM calls go through the existing LiteLLM proxy in the parent stack. Designed to sit alongside the main openwebui-docker deployment and be called via Open WebUI's `RAG_EXTERNAL_RETRIEVAL_API_KEY` mechanism.

## Build & Run

```shell
# Local development (auto-reload)
cp .env.example .env        # then edit API_KEY and connection settings
pip install .
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Or: python app/main.py  (uses settings from .env for host/port, enables reload)

# Docker
docker build -t retrieval-service .
docker run --env-file .env -p 8000:8000 retrieval-service
```

Python 3.11+ required. Dev tools (ruff, pytest) are installed when building with `ENV=dev`.

```shell
# Lint & format
docker compose exec retrieval ruff check .    # lint
docker compose exec retrieval ruff format .   # format

# Tests
docker compose exec retrieval pytest -v       # run all tests
```

## Architecture

Single FastAPI app (`app/main.py`) with endpoints:

- **`POST /search`** — accepts `{queries, collection_names, k}`, returns `{documents, metadatas, distances}`. Bearer-token auth via `API_KEY`.
- **`GET /health`** — healthcheck.

### Agentic Search Pipeline (`app/services/pipeline.py`)

When `ENABLE_AGENTIC_RAG=true`, queries pass through an LLM-driven agent loop (`app/services/agent.py`) that wraps the retrieval pipeline. When disabled, the service falls back to the traditional linear pipeline.

The agent loop:

1. **Query Analysis** — LLM analyzes the incoming query/messages and decides strategy:
   - Simple lookup → single vector search (skip rewriting)
   - Vague/ambiguous → rewrite query before search
   - Complex/multi-part → decompose into independent sub-queries
2. **Query Rewriting** — LLM reformulates queries for better retrieval (e.g. "What did we decide?" → "Budget decision meeting notes Q4 2025")
3. **Query Decomposition** — complex questions split into sub-queries, each retrieved independently
4. **Retrieval** — the existing pipeline exposed as tools the agent calls:
   - Embed via OpenAI-compatible API (`app/services/embedding.py`)
   - Vector search across collections concurrently (`app/services/qdrant.py`)
   - Optional BM25 hybrid fusion via Reciprocal Rank Fusion (`app/services/bm25.py`)
5. **Relevance Grading** — LLM grades retrieved documents. If results are poor, rewrites the query and retries (Corrective RAG pattern, up to `AGENT_MAX_ITERATIONS` attempts)
6. **Optional Reranking** — cross-encoder reranking via OpenAI-compatible `/v1/rerank` API (`app/services/reranker.py`)
7. **Dedup** by MD5 text hash, limit to k

The agent loop is implemented with **PydanticAI** (`app/services/agent.py`): a PydanticAI `Agent` with typed tools (search, rewrite, grade) and Pydantic structured output for routing decisions. Retrieval services (embedding, Qdrant, BM25, reranker) remain as direct custom code exposed as PydanticAI tools — PydanticAI handles only the agentic reasoning, not retrieval itself. Agent LLM calls route through the LiteLLM proxy (`AGENT_API_BASE_URL` defaults to `http://litellm:4000/v1`), making provider switching a config change. When reranking is enabled, initial retrieval fetches `k * INITIAL_RETRIEVAL_MULTIPLIER` candidates. All GPU/LLM services (embedding, reranking, agent) are configured as external OpenAI-compatible APIs — no local models.

### Qdrant Multitenancy Mapping (`app/services/qdrant.py`)

The service replicates Open WebUI's `qdrant_multitenancy.py` collection-routing logic: collection names like `user-memory-*`, `file-*`, `web-search-*`, hex-hash, and knowledge collections are mapped to shared Qdrant collections with tenant filters. This mapping **must stay in sync** with the Open WebUI fork.

## Configuration

All config via environment variables (or `.env` file), loaded by pydantic-settings in `app/config.py`. Key settings:

- `API_KEY` — must match `RAG_EXTERNAL_RETRIEVAL_API_KEY` in Open WebUI
- `EMBEDDING_MODEL` / `EMBEDDING_API_BASE_URL` / `EMBEDDING_API_KEY` — must match Open WebUI's RAG embedding config exactly
- `QDRANT_MULTITENANCY` — must match Open WebUI's Qdrant multitenancy setting
- `ENABLE_HYBRID_SEARCH` / `ENABLE_RERANKING` — toggle optional pipeline stages
- `RERANKER_MODEL` / `RERANKER_API_BASE_URL` / `RERANKER_API_KEY` — OpenAI-compatible reranker endpoint (e.g. `https://embed.itkdev.dk`)
- `ENABLE_AGENTIC_RAG` — toggle agentic mode (when `false`, falls back to traditional linear pipeline)
- `AGENT_MODEL` / `AGENT_API_BASE_URL` / `AGENT_API_KEY` — LLM for the PydanticAI agent loop (defaults to LiteLLM proxy at `http://litellm:4000/v1`; use a fast/cheap model like GPT-4o-mini)
- `AGENT_MAX_ITERATIONS` — max retry iterations for corrective RAG (default: `3`)

## Rules

- **Never read `.env` files.** They contain secrets (API keys, credentials). Use `.env.example` to understand available settings.
- **Always run Python commands inside the Docker container.** `pip install`, `pytest`, `ruff`, and any other project commands must be executed via `docker compose exec retrieval ...` (or `docker compose run` if the container is not running). Never install or run Python tooling on the host.

## Key Constraints

- The embedding model and query prefix **must** be identical to what Open WebUI uses for indexing, or vector search will return garbage.
- The Qdrant collection prefix and multitenancy mapping must mirror Open WebUI's `qdrant_multitenancy.py`. If upstream changes that mapping, this service breaks.
- BM25 hybrid search scrolls entire collections from Qdrant on every query — only viable for small collections.
- Agentic mode adds 2–5x latency and 2–4x token cost per query vs the linear pipeline. Use a small, fast model (e.g. GPT-4o-mini) for agent decisions to keep cost around ~$0.01/query.
- `AGENT_MAX_ITERATIONS` bounds the corrective RAG loop to prevent runaway retries.
- The traditional linear pipeline remains available via `ENABLE_AGENTIC_RAG=false` for latency-sensitive or cost-sensitive use cases.
