import os

# Override env vars BEFORE any app imports (Settings() runs at import time)
os.environ["API_KEY"] = "test-api-key"
os.environ["EMBEDDING_API_BASE_URL"] = "http://fake-embedding:8080"
os.environ["EMBEDDING_API_KEY"] = "fake-key"
os.environ["EMBEDDING_PREFIX_QUERY"] = "query: "
os.environ["QDRANT_URI"] = "http://fake-qdrant:6333"
os.environ["QDRANT_INDEX"] = "ingestion_files"
os.environ["AGENT_API_BASE_URL"] = "http://fake-agent:4000/v1"
os.environ["AGENT_API_KEY"] = "fake-agent-key"
# Force feature flags off by default — individual tests opt in via monkeypatch.
# Without this, the container's compose env (ENABLE_HYBRID_SEARCH=true,
# ENABLE_AGENTIC_RAG=true, …) would leak into the test suite and cause real
# network calls against fake hostnames.
os.environ["ENABLE_HYBRID_SEARCH"] = "false"
os.environ["ENABLE_RERANKING"] = "false"
os.environ["ENABLE_AGENTIC_RAG"] = "false"
os.environ["ENABLE_QUERY_GENERATION"] = "false"
# Sparse embedder defaults to fastembed but tests never let it actually load
# weights — the sparse_embedding module is patched at the call site.
os.environ["SPARSE_QUERY_PROVIDER"] = "none"

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services import bm25, embedding, qdrant, query_generation, reranker, sparse_embedding

# Force settings to match test env. Settings() is instantiated at import time,
# so attribute reassignment is the only reliable way to override fields when the
# container's compose env disagrees with what tests expect.
settings.api_key = "test-api-key"
settings.enable_hybrid_search = False
settings.enable_reranking = False
settings.enable_agentic_rag = False
settings.enable_query_generation = False
settings.sparse_query_provider = "none"


@pytest.fixture
def api_headers():
    return {"Authorization": "Bearer test-api-key"}


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def reset_clients():
    yield
    qdrant._client = None
    qdrant.reset_sparse_capability()
    embedding._client = None
    reranker._client = None
    query_generation._client = None
    sparse_embedding.close()
    bm25.clear_cache()
