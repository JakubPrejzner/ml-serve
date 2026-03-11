from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    label: str
    score: float = Field(..., ge=0.0, le=1.0)
    model_name: str


class PredictResponse(BaseModel):
    result: PredictionResult
    latency_ms: float
    request_id: str


class BatchPredictResponse(BaseModel):
    results: list[PredictionResult]
    latency_ms: float
    request_id: str
    batch_size: int


class ABPredictResponse(PredictResponse):
    variant: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    uptime_seconds: float


class ErrorResponse(BaseModel):
    detail: str
    request_id: str
