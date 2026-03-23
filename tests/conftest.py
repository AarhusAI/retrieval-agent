import os

# Override env vars BEFORE any app imports (Settings() runs at import time)
os.environ["API_KEY"] = "test-api-key"
os.environ["EMBEDDING_API_BASE_URL"] = "http://fake-embedding:8080"
os.environ["EMBEDDING_API_KEY"] = "fake-key"
os.environ["QDRANT_URI"] = "http://fake-qdrant:6333"
os.environ["AGENT_API_BASE_URL"] = "http://fake-agent:4000/v1"
os.environ["AGENT_API_KEY"] = "fake-agent-key"

import pytest
from httpx import ASGITransport, AsyncClient

from app.config import settings
from app.main import app
from app.services import qdrant

# Force settings to match test env (in case .env file or container env overrode them)
settings.api_key = "test-api-key"


@pytest.fixture
def api_headers():
    return {"Authorization": "Bearer test-api-key"}


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture(autouse=True)
def reset_qdrant_client():
    yield
    qdrant._client = None
