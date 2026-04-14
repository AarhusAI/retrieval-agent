from unittest.mock import MagicMock, patch


async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_ready_ok(client):
    mock_client = MagicMock()
    mock_client.get_collections.return_value = MagicMock()
    with patch("app.services.qdrant.get_client", return_value=mock_client):
        resp = await client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_ready_qdrant_down(client):
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = ConnectionError("Connection refused")
    with patch("app.services.qdrant.get_client", return_value=mock_client):
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "error"
