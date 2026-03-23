import pytest


async def test_valid_api_key(client, api_headers):
    resp = await client.get("/health")
    assert resp.status_code == 200


async def test_missing_bearer_token(client):
    resp = await client.post("/search", json={"queries": ["q"], "collection_names": ["c"]})
    assert resp.status_code in (401, 403)


async def test_invalid_bearer_token(client):
    resp = await client.post(
        "/search",
        json={"queries": ["q"], "collection_names": ["c"]},
        headers={"Authorization": "Bearer wrong-key"},
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
    assert resp.status_code in (401, 403)
