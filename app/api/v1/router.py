from fastapi import APIRouter

from app.api.v1.ab_test import router as ab_router
from app.api.v1.health import router as health_router
from app.api.v1.predict import router as predict_router

v1_router = APIRouter(prefix="/v1")

v1_router.include_router(health_router)
v1_router.include_router(predict_router)
v1_router.include_router(ab_router)
