from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Auth
    api_key: str

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

    # Debug
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
