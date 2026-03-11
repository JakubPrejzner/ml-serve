from typing import Any

from fastapi import APIRouter, Depends

from app.api.deps import get_inference_service, get_request_id, get_settings, verify_api_key
from app.config import Settings
from app.schemas.requests import ABPredictRequest
from app.schemas.responses import ABPredictResponse
from app.services.ab_testing import ab_service
from app.services.inference import InferenceService

router = APIRouter(tags=["ab-testing"], dependencies=[Depends(verify_api_key)])


@router.post("/predict/ab", response_model=ABPredictResponse)
async def ab_predict(
    body: ABPredictRequest,
    request_id: str = Depends(get_request_id),
    svc: InferenceService = Depends(get_inference_service),
    settings: Settings = Depends(get_settings),
) -> ABPredictResponse:
    variant = ab_service.assign_variant(settings.ab_test_split)

    # TODO: variant B should eventually point to a challenger model
    model_name = settings.model_name

    try:
        result, latency = svc.predict(body.text, model_name)
    except Exception:
        ab_service.track_result(variant, 0.0, error=True)
        raise

    ab_service.track_result(variant, latency, error=False)

    return ABPredictResponse(
        result=result,
        latency_ms=latency,
        request_id=request_id,
        variant=variant,
    )


@router.get("/ab/results")
async def ab_results() -> dict[str, Any]:
    """Current A/B test metrics. No auth on this one — it's read-only stats."""
    return ab_service.get_results()
