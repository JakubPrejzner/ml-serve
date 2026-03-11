"""
Tests for A/B testing endpoints.
"""

import pytest
import httpx


pytestmark = pytest.mark.anyio


async def test_ab_predict(async_client: httpx.AsyncClient):
    """POST /v1/predict/ab should return a prediction with a variant field."""
    resp = await async_client.post(
        "/v1/predict/ab",
        json={"text": "this is good stuff"},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["variant"] in ("A", "B")
    assert "result" in data
    assert data["result"]["label"] in ("positive", "negative")
    assert "latency_ms" in data
    assert "request_id" in data


async def test_ab_results(async_client: httpx.AsyncClient):
    """
    Fire a handful of predictions through the A/B endpoint,
    then check /v1/ab/results has counted them.
    """
    num_requests = 10
    for _ in range(num_requests):
        resp = await async_client.post(
            "/v1/predict/ab",
            json={"text": "good product"},
        )
        assert resp.status_code == 200

    results_resp = await async_client.get("/v1/ab/results")
    assert results_resp.status_code == 200

    data = results_resp.json()
    variants = data["variants"]

    total = variants["A"]["count"] + variants["B"]["count"]
    assert total == num_requests

    # both variants should have non-negative avg latency
    for v in ("A", "B"):
        assert variants[v]["avg_latency_ms"] >= 0
        assert variants[v]["error_count"] == 0


async def test_ab_split_distribution(async_client: httpx.AsyncClient):
    """
    With a 50/50 split, running 100 requests should give each variant
    roughly 30-70 hits. We're not testing exact 50/50 — just that
    the randomization isn't broken and both sides get traffic.
    """
    counts = {"A": 0, "B": 0}

    for _ in range(100):
        resp = await async_client.post(
            "/v1/predict/ab",
            json={"text": "decent movie"},
        )
        assert resp.status_code == 200
        variant = resp.json()["variant"]
        counts[variant] += 1

    # sanity check: both variants should get at least 20 hits
    # (p(X < 20) for binomial(100, 0.5) is astronomically low)
    assert counts["A"] >= 20, f"Variant A only got {counts['A']} hits out of 100"
    assert counts["B"] >= 20, f"Variant B only got {counts['B']} hits out of 100"


async def test_ab_results_empty(async_client: httpx.AsyncClient):
    """Before any requests, both variant counts should be zero."""
    resp = await async_client.get("/v1/ab/results")
    assert resp.status_code == 200

    data = resp.json()
    assert data["variants"]["A"]["count"] == 0
    assert data["variants"]["B"]["count"] == 0
