"""Metrics instrumentation (app/metrics.py) + the /metrics endpoint.

Collectors are process singletons on the default registry, so assertions use
before/after sample-value deltas — never absolute counts.
"""

import pytest
from prometheus_client import REGISTRY

from app import metrics


async def test_time_stage_records_on_success():
    label = {"stage": "unittest_ok"}
    before = REGISTRY.get_sample_value("retrieval_stage_duration_seconds_count", label) or 0.0
    async with metrics.time_stage("unittest_ok"):
        pass
    after = REGISTRY.get_sample_value("retrieval_stage_duration_seconds_count", label)
    assert after == before + 1


async def test_time_stage_records_and_reraises():
    """A failing stage still contributes its latency, and the exception propagates."""
    label = {"stage": "unittest_err"}
    before = REGISTRY.get_sample_value("retrieval_stage_duration_seconds_count", label) or 0.0
    with pytest.raises(ValueError):
        async with metrics.time_stage("unittest_err"):
            raise ValueError("boom")
    after = REGISTRY.get_sample_value("retrieval_stage_duration_seconds_count", label)
    assert after == before + 1


def test_search_requests_counter_increments():
    label = {"pipeline": "linear", "outcome": "success", "code": "none"}
    before = REGISTRY.get_sample_value("search_requests_total", label) or 0.0
    metrics.search_requests_total.labels(**label).inc()
    after = REGISTRY.get_sample_value("search_requests_total", label)
    assert after == before + 1


async def test_metrics_endpoint_exposes_collectors(client, api_headers):
    response = await client.get("/metrics", headers=api_headers)
    assert response.status_code == 200
    body = response.text
    assert "search_requests_total" in body
    assert "retrieval_stage_duration_seconds" in body


async def test_metrics_endpoint_requires_auth(client):
    """Bearer-protected like /search: no token → rejected, wrong token → 401."""
    no_token = await client.get("/metrics")
    assert no_token.status_code in (401, 403)

    wrong_token = await client.get("/metrics", headers={"Authorization": "Bearer wrong-key"})
    assert wrong_token.status_code == 401


async def test_metrics_endpoint_404_when_disabled(client, api_headers, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "metrics_enabled", False)
    response = await client.get("/metrics", headers=api_headers)
    assert response.status_code == 404
