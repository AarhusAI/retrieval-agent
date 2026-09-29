# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog], and this project adheres to [Semantic Versioning].

## [Unreleased]

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
