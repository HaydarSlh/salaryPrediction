"""
routers/health.py
-----------------
GET /health — liveness check.
Deployment platforms (Render, Railway) ping this to verify the service is up.
"""

from fastapi import APIRouter, Request
from app.schemas import HealthResponse
from app.config import MODEL_VERSION

router = APIRouter(tags=["Monitoring"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
)
def health(request: Request):
    """
    Returns 200 with model status.
    If the predictor failed to load at startup, model_loaded will be False.
    """
    predictor = request.app.state.predictor
    return HealthResponse(
        status="ok",
        model_loaded=predictor is not None,
        model_version=MODEL_VERSION,
    )
