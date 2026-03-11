from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    model_config = {"strict": True}

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["This movie was absolutely fantastic!"],
    )


class BatchPredictRequest(BaseModel):
    model_config = {"strict": True}

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=32,
        examples=[["Great product!", "Terrible experience, would not recommend."]],
    )


class ABPredictRequest(BaseModel):
    model_config = {"strict": True}

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["The food was decent but the service was slow."],
    )
