import logging
from typing import Any

from transformers import pipeline

from app.models.registry import registry

logger = logging.getLogger(__name__)

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"


class SentimentModel:
    def __init__(self) -> None:
        self.pipe: Any = None
        self.name = "sentiment"

    def load(self) -> None:
        logger.info(f"Loading sentiment model: {MODEL_ID}")
        self.pipe = pipeline(
            "text-classification",
            model=MODEL_ID,
            # truncate long inputs instead of crashing
            truncation=True,
        )
        logger.info("Sentiment model loaded")

    def predict(self, text: str) -> dict[str, Any]:
        result = self.pipe(text)[0]
        return {
            "label": result["label"].lower(),
            "score": round(result["score"], 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        results = self.pipe(texts)
        return [
            {"label": r["label"].lower(), "score": round(r["score"], 4)}
            for r in results
        ]

    def warmup(self) -> None:
        """Run a dummy prediction to warm up the pipeline / JIT."""
        logger.info("Warming up sentiment model...")
        self.predict("warmup")
        logger.info("Warmup done")


# register on import
registry.register("sentiment", SentimentModel)
