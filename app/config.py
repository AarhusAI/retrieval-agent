from pydantic import field_validator
from pydantic_settings import BaseSettings

# Placeholder values that historically shipped in docker-compose defaults.
# Treat them as misconfiguration and refuse to boot.
_PLACEHOLDER_API_KEYS = {
    "",
    "change_me",
    "change_me_now",
    "changeme",
    "secret",
    "password",
}
_MIN_API_KEY_LENGTH = 32


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Auth
    api_key: str

    @field_validator("api_key")
    @classmethod
    def _reject_weak_api_key(cls, v: str) -> str:
        normalised = v.strip()
        if normalised.lower() in _PLACEHOLDER_API_KEYS:
            raise ValueError("API_KEY is unset or a known placeholder; set a strong value in .env")
        if len(normalised) < _MIN_API_KEY_LENGTH:
            raise ValueError(f"API_KEY must be at least {_MIN_API_KEY_LENGTH} characters")
        return v

    # Qdrant — single physical collection populated by the ingestion service.
    # The legacy multitenancy mapping (one physical collection per Open WebUI
    # collection class) was retired in Phase 3 alongside the schema change.
    qdrant_uri: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_index: str = "ingestion_files"

    # Embedding (OpenAI-compatible API).
    # embedding_prefix_query must match what the ingestion service used at index
    # time (e5: "query: " on queries / "passage: " on docs; bge-m3: none).
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_prefix_query: str = "query: "

    # Hybrid search. When enabled, retrieval uses native Qdrant hybrid (Query API
    # with prefetch + RRF fusion) for collections that carry sparse vectors, and
    # falls back to client-side BM25 RRF for collections that don't.
    enable_hybrid_search: bool = False
    hybrid_bm25_weight: float = 0.3
    bm25_cache_ttl_seconds: int = 300
    # Hard cap on documents pulled into the in-memory BM25 index per
    # collection scope. Protects the process from OOM when a logical
    # collection has unexpectedly grown.
    bm25_max_docs: int = 10_000

    # Sparse query embedder (used when hybrid is enabled and the configured
    # Qdrant collection has a sparse named vector). Must match the model the
    # ingestion service used for sparse indexing.
    sparse_query_provider: str = "fastembed"  # fastembed | none
    sparse_query_model: str = "Qdrant/bm42-all-minilm-l6-v2-attentions"

    # Reranking (OpenAI-compatible API)
    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_api_base_url: str = ""
    reranker_api_key: str = ""
    # How the cross-encoder ranking combines with the retrieval (dense/hybrid)
    # ranking. "rrf" = Reciprocal Rank Fusion — keeps a strong dense hit that the
    # cross-encoder underranks; "replace" = cross-encoder score only (the older
    # behaviour, where the reranker can bury a top dense hit).
    rerank_fusion: str = "rrf"
    rerank_rrf_k: int = 60
    initial_retrieval_multiplier: int = 3

    # Query generation (from messages, for linear pipeline; agentic uses system prompt)
    enable_query_generation: bool = True
    retrieval_query_generation_prompt_template: str = ""

    # Agentic RAG
    enable_agentic_rag: bool = False
    agent_model: str = "gpt-4o-mini"
    agent_api_base_url: str = "http://litellm:4000/v1"
    agent_api_key: str = ""
    agent_max_iterations: int = 3
    agent_tool_preview_chars: int = 200
    agent_strict_tools: bool = True
    agent_timeout: int = 60
    agent_system_prompt: str = ""
    # Agentic recall (decoupled from request.k to widen recall without
    # blowing the agent LLM's context window).
    agent_fetch_k: int = 20
    agent_preview_k: int = 5
    agent_conversation_history_messages: int = 4
    # Always run a deterministic retrieval pass with the user's original query
    # (seeded before the agent loop) so recall doesn't depend on how the agent
    # rewrites the question. Merged with the agent's retrievals downstream.
    agent_include_raw_query: bool = True

    # ----- Observability -----
    # LOG_LEVEL is the primary verbosity dial, applied to the root logger (so
    # third-party libs follow it too). LOG_FORMAT picks human-readable vs JSON
    # (one object per line, for Loki / a structured-log pipeline). METRICS_ENABLED
    # gates only the GET /metrics endpoint — instrumentation always runs.
    log_level: str = "INFO"
    log_format: str = "text"
    metrics_enabled: bool = True

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"LOG_LEVEL must be one of {sorted(allowed)}; got {v!r}")
        return upper

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        lower = v.lower()
        if lower not in {"text", "json"}:
            raise ValueError(f"LOG_FORMAT must be 'text' or 'json'; got {v!r}")
        return lower

    # Debug — back-compat single switch: bumps the 'app' namespace to DEBUG
    # without flooding third-party loggers. LOG_LEVEL=DEBUG is the broader dial.
    debug: bool = False

    # Server
    host: str = "0.0.0.0"  # nosec B104  # containerized service; binding to all interfaces is intentional
    port: int = 8000


settings = Settings()
