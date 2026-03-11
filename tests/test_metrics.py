"""
Tests for the /metrics Prometheus endpoint.
"""

import pytest
import httpx


pytestmark = pytest.mark.anyio


async def test_metrics_endpoint(async_client: httpx.AsyncClient):
    """GET /metrics should return Prometheus text format."""
    resp = await async_client.get("/metrics")
    assert resp.status_code == 200

    # prometheus text format uses this content type
    assert "text/plain" in resp.headers.get("content-type", "")

    body = resp.text
    # should contain at least some of our custom metrics
    assert "http_requests_total" in body or "http_request_duration_seconds" in body


async def test_metrics_increment(async_client: httpx.AsyncClient):
    """
    Make a prediction, then verify /metrics reflects it.
    The counter for our model + predicted class should bump.
    """
    # fire off a prediction
    predict_resp = await async_client.post(
        "/v1/predict",
        json={"text": "good movie"},
    )
    assert predict_resp.status_code == 200

    # now check metrics
    metrics_resp = await async_client.get("/metrics")
    body = metrics_resp.text

    # should see the inference latency metric for our model
    assert "inference_duration_seconds" in body

    # prediction count should have a "sentiment" label
    assert "model_prediction_total" in body
    assert "sentiment" in body


async def test_metrics_contains_http_counters(async_client: httpx.AsyncClient):
    """After any request, http_requests_total should show up in /metrics."""
    # make any request to generate a counter entry
    await async_client.get("/v1/health")

    resp = await async_client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text
