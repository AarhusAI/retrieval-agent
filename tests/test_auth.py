import pytest


async def test_valid_api_key(client, api_headers):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_missing_bearer_token(client):
    resp = await client.post("/search", json={"queries": ["q"], "collection_names": ["c"]})
    # 401 (not FastAPI's default 403) with a WWW-Authenticate challenge.
    assert resp.status_code == 401
    assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_invalid_bearer_token(client):
    resp = await client.post(
        "/search",
        json={"queries": ["q"], "collection_names": ["c"]},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


async def test_non_ascii_bearer_token_is_401_not_500(client):
    """hmac.compare_digest raises TypeError on non-ASCII str — must stay a 401.

    The header value is sent as latin-1 bytes (httpx refuses non-ASCII str
    values client-side); the server decodes it back to a non-ASCII str.
    """
    resp = await client.post(
        "/search",
        json={"queries": ["q"], "collection_names": ["c"]},
        headers={b"Authorization": "Bearer æøå-ikke-en-nøgle-æøå-ikke".encode("latin-1")},
    )
    assert resp.status_code == 401


@pytest.mark.parametrize(
    "auth_header",
    [
        "Basic dXNlcjpwYXNz",
        "Token test-api-key",
        "",
    ],
)
async def test_non_bearer_schemes_rejected(client, auth_header):
    resp = await client.post(
        "/search",
        json={"queries": ["q"], "collection_names": ["c"]},
        headers={"Authorization": auth_header} if auth_header else {},
    )
    assert resp.status_code == 401
