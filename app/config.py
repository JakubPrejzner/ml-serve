from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "ML_SERVE_"}

    app_name: str = "ml-serve"
    app_version: str = "0.1.0"
    log_level: str = "INFO"

    # model config
    model_name: str = "sentiment"

    # redis / caching
    redis_url: str = "redis://localhost:6379/0"

    # rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # a/b testing
    ab_test_split: float = 0.5

    # auth — comma-separated list, empty means no auth
    api_keys: list[str] = []

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [k for k in v.split(",") if k.strip()] if v.strip() else []
        return v  # type: ignore[return-value]

    # observability
    prometheus_enabled: bool = True


# singleton-ish, import this everywhere
settings = Settings()
