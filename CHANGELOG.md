# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog], and this project adheres to [Semantic Versioning].

## [Unreleased]

### Fixed

- `AGENT_MAX_ITERATIONS` is now enforced as a PydanticAI request limit (iterations + 1) instead of only being stated in
  the system prompt; hitting it returns the partial results gathered so far.
- `BM25_MAX_DOCS`, `RERANK_FUSION`, `RERANK_RRF_K` and `AGENT_INCLUDE_RAW_QUERY` are passed through `docker-compose.yml`
  (previously only picked up via the mounted `.env` in dev).

### Changed

- `RERANK_FUSION` (`rrf`|`replace`) and `SPARSE_QUERY_PROVIDER` (`fastembed`|`none`) are validated at startup; a typo
  no longer silently selects the other branch. `SPARSE_QUERY_PROVIDER` is now case-sensitive.
- Removed `HOST`/`PORT` settings and the unused `python -m app.main` runner; the bind address is set by the uvicorn
  command in the Dockerfile/compose.
- README configuration table lists all settings; CLAUDE.md references `LOG_LEVEL_APP` instead of the removed `DEBUG`.

## [0.0.4] - 2026-10-06

### Added

- GitHub Actions workflows and lint tasks (markdown, YAML).

### Changed

- Converted dependency management from pip to uv.
- Upgraded `pydantic-ai-slim` to `>=2.0,<3.0` (older releases break at import with current OpenTelemetry).
- Blocking Qdrant client calls run in worker threads, so a slow query no longer stalls the event loop (incl. the
  readiness probe).
- Per-query retrieval runs concurrently in the linear pipeline.
- Auth returns 401 with `WWW-Authenticate` on missing/invalid credentials (was 403).
- Query generation honours `AGENT_CONVERSATION_HISTORY_MESSAGES`.
- Simplified app code (about 260 fewer lines): embedding and reranker share one HTTP client, both pipelines share one
  dedup helper (exact-text instead of MD5), the agent's three per-query retrieval loops are one helper, and
  `QdrantResult` is merged into `RetrievalResult`.
- Request validation uses pydantic's built-in `max_length`/`pattern` checks, so 422 messages use pydantic's wording.
- The agent fallback parser takes the outermost `{...}` after the last `[TOOL_CALLS]` marker (or in the whole output
  when absent), covering plain JSON and `[TOOL_CALLS]` text.
- `API_KEY` placeholder list removed; the 32-character minimum already rejects every entry.

### Fixed

- Reranker fails open on all errors (timeouts, protocol errors, malformed or short responses) instead of returning 500.
- Agent `retrieve` tool short-circuits on empty queries instead of calling the embedding API with no input.
- Document metadata is carried through the client-side BM25 fusion path, so BM25-only hits keep their citations.
- BM25 lock-dict leak for collection scopes with no documents.
- Non-ASCII bearer tokens no longer cause a 500.
- Agent httpx client is closed at shutdown and given a timeout aligned with `AGENT_TIMEOUT`.
- `EMBEDDING_API_BASE_URL` is validated at boot, with a clear error on unexpected embedding API response shapes.
- Query text is sanitized in agent debug logs.

## [0.0.3] - 2026-06-11

### Changed

- New `LOG_LEVEL_APP` setting overrides the level of the service's own `app.*` loggers without the third-party
  debug flood.

### Removed

- `DEBUG` setting. Use `LOG_LEVEL_APP=DEBUG` instead.

## [0.0.2] - 2026-06-11

First tagged release.

### Added

- Agentic retrieval service for Open WebUI: FastAPI `POST /search` with a PydanticAI agent loop (query rewriting,
  decomposition, relevance grading with retry) and a linear pipeline fallback.
- Hybrid search: native Qdrant sparse vectors (in-process fastembed) or client-side BM25 fallback.
- Cross-encoder reranking, RRF-fused with the retrieval order (`RERANK_FUSION`).
- Single-collection Qdrant schema written by the external ingestion service, with structural metadata support.
- Prometheus metrics at `GET /metrics` (`METRICS_ENABLED`) and JSON log format (`LOG_FORMAT`).
- Raw-query seed retrieval (`AGENT_INCLUDE_RAW_QUERY`) and round-robin interleave of per-query results.
- Multi-arch (arm) image builds.
- Mozilla Public License 2.0.

### Security

- Chat content and query strings are only logged at debug level (GDPR).
- `SearchRequest` fields are bounded to prevent cost and memory amplification.
- Removed weak default credentials; hardened container runtime and pinned Debian base.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[unreleased]: https://github.com/AarhusAI/retrieval-agent/compare/0.0.3...HEAD
[0.0.3]: https://github.com/AarhusAI/retrieval-agent/releases/tag/0.0.3
[0.0.2]: https://github.com/AarhusAI/retrieval-agent/releases/tag/0.0.2
