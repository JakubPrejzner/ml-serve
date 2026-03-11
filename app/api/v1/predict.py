from fastapi import APIRouter, Depends

from app.api.deps import get_inference_service, get_settings, verify_api_key, get_request_id
from app.schemas.requests import PredictRequest, BatchPredictRequest
from app.schemas.responses import PredictResponse, BatchPredictResponse
from app.services.inference import InferenceService
from app.config import Settings
from app.middleware.metrics import INFERENCE_LATENCY, MODEL_PREDICTION_COUNT, BATCH_SIZE

router = APIRouter(tags=["inference"], dependencies=[Depends(verify_api_key)])


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    request_id: str = Depends(get_request_id),
    svc: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_settings),
):
    result, latency = svc.predict(body.text, settings.model_name)

    # track prometheus stuff
    INFERENCE_LATENCY.labels(model=settings.model_name).observe(latency / 1000)
    MODEL_PREDICTION_COUNT.labels(
        model=settings.model_name, predicted_class=result.label
    ).inc()

    return PredictResponse(
        result=result,
        latency_ms=latency,
        request_id=request_id,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    body: BatchPredictRequest,
    request_id: str = Depends(get_request_id),
    svc: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_settings),
):
    results, latency = svc.predict_batch(body.texts, settings.model_name)

    INFERENCE_LATENCY.labels(model=settings.model_name).observe(latency / 1000)
    BATCH_SIZE.observe(len(body.texts))
    for r in results:
        MODEL_PREDICTION_COUNT.labels(
            model=settings.model_name, predicted_class=r.label
        ).inc()

    return BatchPredictResponse(
        results=results,
        latency_ms=latency,
        request_id=request_id,
        batch_size=len(body.texts),
    )
