import time

from fastapi import APIRouter
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ── counters ──────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

MODEL_PREDICTION_COUNT = Counter(
    "model_prediction_total",
    "Predictions per model and class",
    ["model", "predicted_class"],
)

RATE_LIMIT_HITS = Counter(
    "rate_limit_hits_total",
    "Total requests rejected by rate limiter",
)

# ── histograms ────────────────────────────────────────────

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

INFERENCE_LATENCY = Histogram(
    "inference_duration_seconds",
    "Model inference latency",
    ["model"],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

BATCH_SIZE = Histogram(
    "batch_request_size",
    "Number of texts per batch request",
    buckets=[1, 2, 4, 8, 16, 32],
)

# ── gauges ────────────────────────────────────────────────

ACTIVE_REQUESTS = Gauge(
    "http_active_requests",
    "Currently in-flight requests",
)


# ── middleware ────────────────────────────────────────────

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # don't instrument the metrics endpoint itself
        if request.url.path == "/metrics":
            return await call_next(request)

        ACTIVE_REQUESTS.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        finally:
            ACTIVE_REQUESTS.dec()

        elapsed = time.perf_counter() - start

        # fixme: high-cardinality path params could blow up label space,
        # but for v1 routes it's fine since they're all fixed paths
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(elapsed)

        return response


# ── /metrics endpoint ─────────────────────────────────────

metrics_router = APIRouter()


@metrics_router.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    body = generate_latest()
    return Response(content=body, media_type=CONTENT_TYPE_LATEST)
