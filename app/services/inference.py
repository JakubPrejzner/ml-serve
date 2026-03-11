import time
import logging

from app.models.registry import registry
from app.schemas.responses import PredictionResult

logger = logging.getLogger(__name__)


class InferenceError(Exception):
    pass


class ModelNotFoundError(InferenceError):
    pass


class InferenceService:
    def predict(self, text: str, model_name: str) -> tuple[PredictionResult, float]:
        try:
            model = registry.get(model_name)
        except KeyError as exc:
            raise ModelNotFoundError(str(exc)) from exc

        start = time.perf_counter()
        try:
            raw = model.predict(text)
        except Exception as exc:
            logger.error(f"Inference failed for model={model_name}: {exc}")
            raise InferenceError(f"Inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        result = PredictionResult(
            label=raw["label"],
            score=raw["score"],
            model_name=model_name,
        )
        return result, round(latency_ms, 2)

    def predict_batch(
        self, texts: list[str], model_name: str
    ) -> tuple[list[PredictionResult], float]:
        try:
            model = registry.get(model_name)
        except KeyError as exc:
            raise ModelNotFoundError(str(exc)) from exc

        start = time.perf_counter()
        try:
            raw_results = model.predict_batch(texts)
        except Exception as exc:
            logger.error(f"Batch inference failed for model={model_name}: {exc}")
            raise InferenceError(f"Batch inference failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        results = [
            PredictionResult(
                label=r["label"],
                score=r["score"],
                model_name=model_name,
            )
            for r in raw_results
        ]
        return results, round(latency_ms, 2)


# single instance, no need for multiple
inference_service = InferenceService()
