from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # Auth
    api_key: str

    # Qdrant
    qdrant_uri: str = "http://qdrant:6333"
    qdrant_api_key: str | None = None
    qdrant_collection_prefix: str = "open-webui"
    qdrant_multitenancy: bool = True

    # Embedding (OpenAI-compatible API)
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_api_base_url: str = ""
    embedding_api_key: str = ""
    embedding_query_prefix: str = "query: "

    # Hybrid search
    enable_hybrid_search: bool = False
    hybrid_bm25_weight: float = 0.3

    # Reranking (OpenAI-compatible API)
    enable_reranking: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_api_base_url: str = ""
    reranker_api_key: str = ""
    initial_retrieval_multiplier: int = 3

    # Agentic RAG
    enable_agentic_rag: bool = False
    agent_model: str = "gpt-4o-mini"
    agent_api_base_url: str = "http://litellm:4000/v1"
    agent_api_key: str = ""
    agent_max_iterations: int = 3
    agent_tool_preview_chars: int = 200

    # Server
    host: str = "0.0.0.0"
    port: int = 8000


settings = Settings()
