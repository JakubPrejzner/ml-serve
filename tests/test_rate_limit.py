"""
Tests for the token-bucket rate limiter.

We override the module-level settings in the rate_limit module and rebuild
the middleware stack to get fresh buckets with predictable capacity.
"""

import httpx
import pytest

from app.api.deps import get_inference_service, get_settings
from app.models.registry import registry
from app.services.inference import InferenceService
from tests.conftest import FakeModel, _test_settings

pytestmark = pytest.mark.anyio


@pytest.fixture()
def _inject_model():
    """Put fake model into registry for rate limit tests (separate from async_client)."""
    model = FakeModel()
    old_instances = dict(registry._instances)
    old_registry = dict(registry._registry)

    registry._registry["sentiment"] = FakeModel
    registry._instances["sentiment"] = model

    yield

    registry._instances = old_instances
    registry._registry = old_registry


async def _make_rate_limited_client(rate_limit_requests: int = 3):
    """
    Build a fresh AsyncClient where the rate limiter uses the given capacity.
    Patches the module-level settings in rate_limit.py and forces a middleware
    stack rebuild so a new RateLimitMiddleware instance picks up the new values.
    """
    import app.middleware.rate_limit as rl_mod
    from app.main import app

    settings = _test_settings(
        rate_limit_requests=rate_limit_requests,
        rate_limit_window=60,
    )

    # the middleware reads this directly — DI overrides won't help here
    rl_mod.settings = settings
    app.middleware_stack = None  # new middleware = new buckets

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_inference_service] = lambda: InferenceService()

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )
    return client, app


async def test_under_limit(_inject_model):
    """Making fewer requests than the limit — all should be 200."""
    client, app = await _make_rate_limited_client(rate_limit_requests=10)

    try:
        for _ in range(5):
            resp = await client.post("/v1/predict", json={"text": "good vibes"})
            assert resp.status_code == 200
    finally:
        app.dependency_overrides.clear()
        await client.aclose()


async def test_over_limit(_inject_model):
    """
    Bucket capacity = 3. The first 3 requests drain it,
    then the next one should come back 429 with Retry-After.
    """
    client, app = await _make_rate_limited_client(rate_limit_requests=3)

    try:
        # drain the bucket
        for i in range(3):
            resp = await client.post("/v1/predict", json={"text": "whatever"})
            assert resp.status_code == 200, f"request {i + 1} should pass"

        # this one should be rejected
        resp = await client.post("/v1/predict", json={"text": "one too many"})
        assert resp.status_code == 429
        assert "retry-after" in resp.headers
    finally:
        app.dependency_overrides.clear()
        await client.aclose()


async def test_different_clients(_inject_model):
    """
    Two different API keys get independent rate limit buckets.
    Exhausting one shouldn't block the other.
    """
    client, app = await _make_rate_limited_client(rate_limit_requests=3)

    try:
        # drain client A's bucket (3 tokens + 1 over)
        for _ in range(4):
            await client.post(
                "/v1/predict",
                json={"text": "text"},
                headers={"X-API-Key": "client-a-key"},
            )

        # client A is spent
        resp_a = await client.post(
            "/v1/predict",
            json={"text": "text"},
            headers={"X-API-Key": "client-a-key"},
        )

        # client B has a fresh bucket
        resp_b = await client.post(
            "/v1/predict",
            json={"text": "text"},
            headers={"X-API-Key": "client-b-key"},
        )

        assert resp_a.status_code == 429, "client A should be rate limited"
        assert resp_b.status_code == 200, "client B should still have tokens"
    finally:
        app.dependency_overrides.clear()
        await client.aclose()
