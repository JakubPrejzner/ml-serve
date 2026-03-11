import threading
import time
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.config import settings
from app.middleware.metrics import RATE_LIMIT_HITS


class _TokenBucket:
    """Classic token-bucket. One per client key."""

    __slots__ = ("capacity", "tokens", "refill_rate", "last_refill")

    def __init__(self, capacity: int, refill_rate: float) -> None:
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens/sec
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self) -> bool:
        self._refill()
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    def retry_after(self) -> int:
        """Seconds until at least 1 token is available."""
        self._refill()
        if self.tokens >= 1.0:
            return 0
        wait = (1.0 - self.tokens) / self.refill_rate
        return int(wait) + 1  # round up so the client doesn't retry too early


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Per-client token bucket rate limiter.
    Keys on X-API-Key if present, otherwise falls back to client IP.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self._buckets: dict[str, _TokenBucket] = {}
        self._lock = threading.Lock()

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key}"
        host = request.client.host if request.client else "unknown"
        return f"ip:{host}"

    def _get_bucket(self, key: str) -> _TokenBucket:
        with self._lock:
            if key not in self._buckets:
                window = max(settings.rate_limit_window, 1)
                self._buckets[key] = _TokenBucket(
                    capacity=settings.rate_limit_requests,
                    refill_rate=settings.rate_limit_requests / window,
                )
            return self._buckets[key]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # skip health checks and metrics
        path = request.url.path
        if "/health" in path or path == "/metrics":
            return await call_next(request)

        bucket = self._get_bucket(self._client_key(request))

        if not bucket.consume():
            RATE_LIMIT_HITS.inc()
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Slow down."},
                headers={"Retry-After": str(bucket.retry_after())},
            )

        return await call_next(request)
