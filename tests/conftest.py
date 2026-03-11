"""
Shared fixtures for the ml-serve test suite.
Everything runs without Docker, Redis, or a real HuggingFace model.
"""

import pytest
import httpx

from app.config import Settings
from app.services.inference import InferenceService
from app.api.deps import get_settings, get_inference_service
from app.models.registry import registry
from app.services.ab_testing import ab_service


# ── helpers ──────────────────────────────────────────────────


class FakeModel:
    """
    Drop-in replacement for SentimentModel.
    Returns deterministic results so tests don't need HuggingFace.
    """

    name = "sentiment"

    def load(self):
        pass

    def warmup(self):
        pass

    def predict(self, text: str) -> dict:
        # positive for anything containing "good" or "great", negative otherwise
        label = "positive" if any(w in text.lower() for w in ("good", "great", "fantastic")) else "negative"
        return {"label": label, "score": 0.9876}

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


def _test_settings(**overrides) -> Settings:
    defaults = dict(
        rate_limit_requests=1000,
        rate_limit_window=60,
        api_keys=[],
        prometheus_enabled=True,
        model_name="sentiment",
        log_level="WARNING",
    )
    defaults.update(overrides)
    return Settings(**defaults)


# ── fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def test_settings():
    """Settings with generous rate limits and no auth by default."""
    return _test_settings()


@pytest.fixture()
def mock_model():
    """Fake model that returns canned results instantly."""
    return FakeModel()


@pytest.fixture()
def _inject_mock_model(mock_model):
    """
    Shove the fake model into the registry so InferenceService
    picks it up instead of trying to download DistilBERT.
    """
    old_instances = dict(registry._instances)
    old_registry = dict(registry._registry)

    registry._registry["sentiment"] = FakeModel
    registry._instances["sentiment"] = mock_model

    yield mock_model

    registry._instances = old_instances
    registry._registry = old_registry


@pytest.fixture(autouse=True)
def _fresh_rate_limiter():
    """
    The rate limiter middleware reads the module-level `settings` object
    directly (not through DI), and caches token buckets per client.
    We patch it to a very high limit and force a middleware stack rebuild
    before each test so every test starts with a fresh set of buckets.
    """
    import app.middleware.rate_limit as rl_mod
    from app.main import app

    original_settings = rl_mod.settings
    rl_mod.settings = _test_settings(rate_limit_requests=100_000)
    app.middleware_stack = None

    yield

    rl_mod.settings = original_settings
    app.middleware_stack = None


@pytest.fixture()
async def async_client(test_settings, _inject_mock_model):
    """
    httpx AsyncClient wired to our FastAPI app.
    Overrides deps so we get test settings + the fake model.
    """
    from app.main import app

    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_inference_service] = lambda: InferenceService()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def _reset_ab_service():
    """Reset A/B counters between tests so they don't bleed through."""
    ab_service.reset()
    yield
    ab_service.reset()
