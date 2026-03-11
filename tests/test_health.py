"""
Tests for health / readiness / liveness probes.
"""

import pytest
import httpx


pytestmark = pytest.mark.anyio


async def test_health_endpoint(async_client: httpx.AsyncClient):
    """GET /v1/health returns status=ok and expected fields."""
    resp = await async_client.get("/v1/health")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "model_loaded" in data
    assert "uptime_seconds" in data
    assert isinstance(data["uptime_seconds"], (int, float))


async def test_readiness(async_client: httpx.AsyncClient):
    """
    Readiness probe should return 200 when model is loaded.
    Our fixture injects the mock model into the registry,
    so it should report as loaded.
    """
    resp = await async_client.get("/v1/health/ready")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "ready"


async def test_liveness(async_client: httpx.AsyncClient):
    """Liveness probe — if the process is up, this returns 200."""
    resp = await async_client.get("/v1/health/live")
    assert resp.status_code == 200

    data = resp.json()
    assert data["status"] == "alive"


async def test_health_shows_model_loaded(async_client: httpx.AsyncClient):
    """model_loaded should be True since we injected the fake model."""
    resp = await async_client.get("/v1/health")
    assert resp.status_code == 200
    assert resp.json()["model_loaded"] is True
