from functools import lru_cache

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.services.inference import InferenceService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_inference_service() -> InferenceService:
    return InferenceService()


async def verify_api_key(
    key: str | None = Security(api_key_header),
    settings: Settings = Depends(get_settings),
) -> str | None:
    # no keys configured = wide open, fine for local dev
    if not settings.api_keys:
        return None
    if key is None or key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


def get_request_id(request: Request) -> str:
    """Grab request_id set by the logging middleware, or fall back."""
    return getattr(request.state, "request_id", "unknown")
