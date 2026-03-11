from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import v1_router
from app.config import settings
from app.middleware.logging import RequestLoggingMiddleware, configure_logging
from app.middleware.metrics import MetricsMiddleware, metrics_router
from app.middleware.rate_limit import RateLimitMiddleware
from app.models import sentiment as _sentiment_init  # noqa: F401
from app.services.inference import InferenceError, ModelNotFoundError

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # ── startup ───────────────────────────────
    configure_logging(settings.log_level)
    logger.info("starting", app=settings.app_name, version=settings.app_version)

    from app.models.registry import registry

    model = registry.get(settings.model_name)
    model.warmup()

    logger.info("ready to serve")
    yield
    # ── shutdown ──────────────────────────────
    logger.info("shutting down")


app = FastAPI(
    title="ml-serve",
    version=settings.app_version,
    description="Production ML inference microservice",
    lifespan=lifespan,
)

# ── middleware stack ──────────────────────────────────────
# added in reverse order: last added = outermost = runs first
# request flow: CORS → Metrics → RateLimit → Logging → handler
#
# metrics wraps rate_limit so we capture 429s in prometheus
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
if settings.prometheus_enabled:
    app.add_middleware(MetricsMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── routes ────────────────────────────────────────────────
app.include_router(v1_router)

if settings.prometheus_enabled:
    app.include_router(metrics_router)


# ── exception handlers ────────────────────────────────────


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(
    request: Request, exc: ModelNotFoundError
) -> JSONResponse:
    rid: Any = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=404,
        content={"detail": str(exc), "request_id": rid},
    )


@app.exception_handler(InferenceError)
async def inference_error_handler(request: Request, exc: InferenceError) -> JSONResponse:
    rid: Any = getattr(request.state, "request_id", "")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "request_id": rid},
    )
