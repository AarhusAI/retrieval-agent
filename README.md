# Agentic Retrieval Service

Agentic retrieval microservice for [Open WebUI](https://github.com/open-webui/open-webui). Wraps RAG document search in
an LLM-driven reasoning loop — adding query rewriting, decomposition, and corrective relevance grading with retry. Falls
back to a traditional linear pipeline when agentic mode is disabled.

Uses [PydanticAI](https://ai.pydantic.dev/) for the agent loop. Retrieval services (embedding, Qdrant, BM25, reranking)
remain as direct custom code exposed as PydanticAI tools. Agent LLM calls route through the existing LiteLLM proxy in
the parent stack.

Integrates with Open WebUI via its external retrieval API — configure `RAG_EXTERNAL_RETRIEVAL_API_KEY` and point it at
this service.

## Requirements

- Docker and Docker Compose
- [Task](https://taskfile.dev/) (Go Task runner)
- Access to a Qdrant instance (shared with Open WebUI)
- An OpenAI-compatible embedding endpoint matching Open WebUI's RAG config
- *(For agentic mode or query generation)* An LLM endpoint (e.g. LiteLLM proxy)

## Quick Start

### Local Development (Docker Compose + [Task](https://taskfile.dev/))

```shell
cp .env.example .env
# Edit .env — at minimum set API_KEY and verify QDRANT_URI and embedding settings

# Generate a secure API key:
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Set the output as API_KEY in .env and as RAG_EXTERNAL_RETRIEVAL_API_KEY in Open WebUI

task setup          # starts containers + installs dev deps (requires Traefik 'frontend' network)
task logs           # tail retrieval container logs
```

Common task commands:

```shell
task up             # start containers
task down           # stop containers
task shell          # open bash shell in the retrieval container
task install        # reinstall deps (pip install '.[dev]')
task lint           # run all linters (ruff check + format --check)
task lint:fix       # auto-fix lint issues
task test           # run all tests (pytest -v)
task test:coverage  # run tests with coverage report
task ci             # lint + test
```

Run a single test:

```shell
docker compose exec retrieval pytest tests/services/test_agent.py -v
docker compose exec retrieval pytest tests/services/test_agent.py::TestAgenticSearch::test_empty_queries -v
```

### Production Image

```shell
task build:image              # build + push to ghcr.io/aarhusai/retrieval-agent:latest
task build:image TAG=v1.0.0   # with specific tag
```

### Standalone Docker (without Compose)

```shell
docker build -t agentic-retrieval .
docker run --env-file .env -p 8000:8000 agentic-retrieval
```

The service exposes two health endpoints:

- `GET /health` — liveness probe (always returns 200 if the process is running)
- `GET /health/ready` — readiness probe (verifies Qdrant connectivity, returns 503 if unreachable)

## API

### `POST /search`

Requires `Authorization: Bearer <API_KEY>` header.

**Request (with explicit queries):**

```json
{
  "queries": [
    "what is the policy on remote work?"
  ],
  "collection_names": [
    "file-abc123",
    "knowledge-base"
  ],
  "k": 5
}
```

**Request (with chat messages — the service generates optimized queries):**

```json
{
  "messages": [
    { "role": "user", "content": "Tell me about remote work" },
    { "role": "assistant", "content": "We have several policies..." },
    { "role": "user", "content": "What about the approval process?" }
  ],
  "collection_names": [
    "file-abc123",
    "knowledge-base"
  ],
  "k": 5,
  "retrieval_query_generation_prompt_template": "optional custom template override"
}
```

At least one of `queries` or `messages` must be provided. When `messages` is given, the service extracts or generates
search queries from the conversation (via LLM when `ENABLE_QUERY_GENERATION=true` or in agentic mode, otherwise falls
back to the last user message).

**Response:**

```json
{
  "documents": [
    [
      "Document text 1",
      "Document text 2"
    ]
  ],
  "metadatas": [
    [
      {
        "source": "..."
      },
      {
        "source": "..."
      }
    ]
  ],
  "distances": [
    [
      0.87,
      0.82
    ]
  ]
}
```

Each top-level list element corresponds to one query. `distances` are normalized to `[0, 1]` (higher = more similar).

## Configuration

All settings are environment variables (or `.env` file). See [`.env.example`](.env.example) for the full list.

| Variable                       | Default                                 | Description                                                                            |
|--------------------------------|-----------------------------------------|----------------------------------------------------------------------------------------|
| `API_KEY`                      | *(required)*                            | Must match `RAG_EXTERNAL_RETRIEVAL_API_KEY` in Open WebUI                              |
| `QDRANT_URI`                   | `http://qdrant:6333`                    | Qdrant connection URL                                                                  |
| `QDRANT_API_KEY`               |                                         | Qdrant API key (if authentication is enabled)                                          |
| `QDRANT_MULTITENANCY`          | `true`                                  | Must match Open WebUI's multitenancy setting                                           |
| `QDRANT_COLLECTION_PREFIX`     | `open-webui`                            | Must match Open WebUI's collection prefix                                              |
| `EMBEDDING_MODEL`              | `intfloat/multilingual-e5-large`        | Must match Open WebUI's RAG embedding model                                            |
| `EMBEDDING_API_BASE_URL`       |                                         | OpenAI-compatible embedding endpoint                                                   |
| `EMBEDDING_API_KEY`            |                                         | API key for embedding endpoint                                                         |
| `EMBEDDING_QUERY_PREFIX`       | `query: `                               | Prefix applied to queries before embedding                                             |
| `ENABLE_HYBRID_SEARCH`         | `false`                                 | Enable BM25 + vector Reciprocal Rank Fusion                                            |
| `HYBRID_BM25_WEIGHT`           | `0.3`                                   | BM25 weight in fusion (vector weight = 1 - this)                                       |
| `BM25_CACHE_TTL_SECONDS`       | `300`                                   | TTL for in-memory BM25 index cache (avoids re-scrolling Qdrant per query)              |
| `ENABLE_RERANKING`             | `false`                                 | Enable cross-encoder reranking stage                                                   |
| `RERANKER_MODEL`               | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model for reranking                                                      |
| `RERANKER_API_BASE_URL`        |                                         | OpenAI-compatible reranker endpoint (e.g. `https://embed.itkdev.dk`)                   |
| `RERANKER_API_KEY`             |                                         | API key for reranker endpoint                                                          |
| `INITIAL_RETRIEVAL_MULTIPLIER` | `3`                                     | Fetch k × multiplier candidates before reranking                                       |
| `ENABLE_QUERY_GENERATION`      | `true`                                  | Enable LLM-based query generation from chat messages (linear pipeline)                 |
| `ENABLE_AGENTIC_RAG`           | `false`                                 | Enable agentic mode (LLM-driven query analysis, rewriting, grading)                    |
| `AGENT_MODEL`                  | `gpt-4o-mini`                           | LLM model for agent decisions and query generation                                     |
| `AGENT_API_BASE_URL`           | `http://litellm:4000/v1`                | LLM endpoint for agent (defaults to LiteLLM proxy)                                     |
| `AGENT_API_KEY`                |                                         | API key for agent LLM endpoint                                                         |
| `AGENT_MAX_ITERATIONS`         | `3`                                     | Max corrective RAG retry iterations                                                    |
| `AGENT_TOOL_PREVIEW_CHARS`     | `200`                                   | Max chars of document text sent to agent LLM for grading (full text stored separately) |
| `AGENT_STRICT_TOOLS`           | `true`                                  | Enable strict tool definition validation in PydanticAI (disable for models that don't support it) |
| `AGENT_TIMEOUT`                | `60`                                    | Wall-clock timeout (seconds) for the agent run — returns partial results on timeout    |
| `AGENT_SYSTEM_PROMPT`          |                                         | Override the default agent system prompt (uses built-in prompt when empty)              |
| `DEBUG`                        | `false`                                 | Enable debug logging (includes per-step token usage for agent and query generation)    |
| `HOST`                         | `0.0.0.0`                               | Server bind address                                                                    |
| `PORT`                         | `8000`                                  | Server port                                                                            |

### Critical: Keeping Settings in Sync

The following settings **must** match your Open WebUI deployment exactly, or search results will be incorrect:

- **Embedding model and query prefix** — mismatched embeddings produce meaningless similarity scores
- **Qdrant collection prefix and multitenancy mode** — the service replicates Open WebUI's collection-routing logic
  internally
- **Qdrant instance** — must point at the same Qdrant that Open WebUI writes to

## Search Pipeline

### Linear Mode (`ENABLE_AGENTIC_RAG=false`)

```mermaid
flowchart LR
    A[POST /search] --> A1{Query\ngeneration\nenabled?}
    A1 -- "yes + messages" --> A2[LLM: Generate\noptimized queries]
    A1 -- no --> A3[Use explicit queries\nor last user message]
    A2 --> B[Embed queries]
    A3 --> B
    B --> C[Vector search\nQdrant]
    C --> D{Hybrid\nenabled?}
    D -- yes --> E[BM25 search +\nRank Fusion]
    D -- no --> F{Reranking\nenabled?}
    E --> F
    F -- yes --> G[Cross-encoder\nrerank]
    F -- no --> H[Dedup + limit k]
    G --> H
    H --> I[Response]
```

1. **Query resolution** — when `messages` are provided and `ENABLE_QUERY_GENERATION=true`, an LLM generates optimized
   retrieval queries from the conversation. Otherwise uses explicit `queries` or falls back to the last user message.
2. **Embed** queries using the configured embedding API
3. **Vector search** across all requested collections concurrently via Qdrant
4. **Hybrid fusion** *(optional)*: BM25 keyword search fused with vector results using Reciprocal Rank Fusion
5. **Reranking** *(optional)*: cross-encoder model rescores the top candidates
6. **Dedup** by content hash, limit to `k` results

### Agentic Mode (`ENABLE_AGENTIC_RAG=true`)

```mermaid
flowchart TD
    A[POST /search] --> B[Extract queries\nfrom request/messages]
    B --> C[LLM: Analyze query]
    C --> D{Strategy?}
    D -- direct --> E[Use query as-is]
    D -- rewrite --> F[LLM: Rewrite query\nfor clarity]
    D -- decompose --> G[LLM: Split into\n2-3 sub-queries]
    E --> H[retrieve tool]
    F --> H
    G --> H

    subgraph retrieve ["retrieve tool (PydanticAI)"]
        H1[Embed queries] --> H2[Vector search\nQdrant]
        H2 --> H3{Hybrid?}
        H3 -- yes --> H4[BM25 + Rank Fusion]
        H3 -- no --> H5{Rerank?}
        H4 --> H5
        H5 -- yes --> H6[Cross-encoder rerank]
        H5 -- no --> H7[Return results]
        H6 --> H7
    end

    H --> H1
    H7 --> I[Store full results\nin side-channel]
    I --> J[Send truncated previews\nto LLM]
    J --> K[LLM: Grade relevance]
    K --> L{Results\nrelevant?}
    L -- yes --> M[Dedup + limit k]
    L -- no --> N{Retries\n< max?}
    N -- yes --> F
    N -- no --> M
    M --> O[Response\nfull text from side-channel]
```

1. **Query Analysis** — LLM decides strategy: direct search, rewrite for clarity, or decompose into sub-queries
2. **Query Rewriting** — LLM reformulates vague queries for better retrieval
3. **Query Decomposition** — complex questions split into independent sub-queries
4. **Retrieval** — existing pipeline (embed → vector search → optional hybrid/rerank) exposed as a PydanticAI tool
5. **Side-channel** — full document text stored in deps; only truncated previews (`AGENT_TOOL_PREVIEW_CHARS`) sent to
   the LLM to reduce token cost
6. **Relevance Grading** — LLM grades truncated results; if poor, rewrites query and retries (up to
   `AGENT_MAX_ITERATIONS`)
7. **Dedup** by content hash, limit to `k` results — response uses full text from the side-channel

### Notes

- BM25 hybrid search scrolls entire collections from Qdrant — only practical for small collections. Results are cached
  in-memory with a TTL (`BM25_CACHE_TTL_SECONDS`, default 5 min) to avoid re-scrolling on every query.
- When reranking is enabled, the initial retrieval fetches `k × INITIAL_RETRIEVAL_MULTIPLIER` candidates to give the
  reranker a larger pool.
- Agentic mode adds 2–5× latency and 2–4× token cost per query. Use a fast, cheap model (e.g. GPT-4o-mini) for agent
  decisions.
- Agent LLM calls default to the LiteLLM proxy at `http://litellm:4000/v1`, making provider switching a config change.
