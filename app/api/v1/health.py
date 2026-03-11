import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.api.deps import get_settings
from app.config import Settings
from app.models.registry import registry
from app.schemas.responses import HealthResponse

router = APIRouter(tags=["health"])

_started_at = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        model_loaded=registry.is_loaded(settings.model_name),
        uptime_seconds=round(time.time() - _started_at, 1),
    )


@router.get("/health/ready", response_model=None)
async def readiness_probe(
    settings: Settings = Depends(get_settings),
) -> JSONResponse | dict[str, str]:
    """Returns 503 until the model is actually loaded and ready."""
    if not registry.is_loaded(settings.model_name):
        return JSONResponse(status_code=503, content={"status": "not ready"})
    return {"status": "ready"}


@router.get("/health/live")
async def liveness_probe() -> dict[str, str]:
    # dead simple — if this responds, the process is alive
    return {"status": "alive"}
