"""
Tests for the /v1/predict and /v1/predict/batch endpoints.
Verifies single prediction, batch, validation rules, etc.
"""

import pytest
import httpx


pytestmark = pytest.mark.anyio


async def test_predict_single(async_client: httpx.AsyncClient):
    """Basic happy-path: send text, get a prediction back."""
    resp = await async_client.post(
        "/v1/predict",
        json={"text": "This product is great"},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert "result" in data
    assert data["result"]["label"] in ("positive", "negative")
    assert 0.0 <= data["result"]["score"] <= 1.0
    assert data["result"]["model_name"] == "sentiment"
    assert "latency_ms" in data
    assert "request_id" in data


async def test_predict_empty_text(async_client: httpx.AsyncClient):
    """Empty string should fail validation (min_length=1)."""
    resp = await async_client.post(
        "/v1/predict",
        json={"text": ""},
    )
    assert resp.status_code == 422

    body = resp.json()
    # FastAPI puts validation errors under "detail"
    assert "detail" in body


async def test_predict_batch(async_client: httpx.AsyncClient):
    """Batch endpoint: send a few texts and check the response shape."""
    texts = ["good stuff", "terrible", "great movie"]
    resp = await async_client.post(
        "/v1/predict/batch",
        json={"texts": texts},
    )
    assert resp.status_code == 200

    data = resp.json()
    assert data["batch_size"] == len(texts)
    assert len(data["results"]) == len(texts)

    for result in data["results"]:
        assert result["label"] in ("positive", "negative")
        assert 0.0 <= result["score"] <= 1.0
        assert result["model_name"] == "sentiment"


async def test_predict_batch_too_large(async_client: httpx.AsyncClient):
    """Exceeding the 32-text limit should give 422."""
    texts = [f"text number {i}" for i in range(33)]
    resp = await async_client.post(
        "/v1/predict/batch",
        json={"texts": texts},
    )
    assert resp.status_code == 422


async def test_predict_missing_body(async_client: httpx.AsyncClient):
    """No JSON body at all -> 422."""
    resp = await async_client.post("/v1/predict")
    assert resp.status_code == 422


async def test_predict_returns_positive_for_good_text(async_client: httpx.AsyncClient):
    """Our fake model is wired to return positive for text containing 'good'."""
    resp = await async_client.post(
        "/v1/predict",
        json={"text": "this is good"},
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["label"] == "positive"
