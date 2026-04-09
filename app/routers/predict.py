"""
routers/predict.py
------------------
POST /predict — main prediction endpoint.
Validates input via Pydantic, delegates to SalaryPredictor service,
returns structured response.
"""

from fastapi import APIRouter, HTTPException, Request
import logging

from app.schemas import PredictionRequest, PredictionResponse
from app.config import MODEL_VERSION

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict annual salary",
)
def predict(request: Request, body: PredictionRequest):
    """
    Accepts data science job details and returns the predicted
    annual salary in USD.

    **Validation rules:**
    - `experience_level`: EN | MI | SE | EX
    - `employment_type`: FT | PT | CT | FL
    - `job_title_bucket`: one of 5 defined buckets
    - `remote_ratio`: 0 | 50 | 100
    - `company_location` / `employee_residence`: 2-letter ISO country code
    - `company_size`: S | M | L
    - `work_year`: 2020–2030
    """
    predictor = request.app.state.predictor

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please try again shortly.",
        )

    try:
        result = predictor.predict(body.model_dump())

    except ValueError as e:
        # Known encoding errors — bad input that slipped past Pydantic
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred during prediction.",
        )

    return PredictionResponse(
        predicted_salary_usd=result["predicted_salary_usd"],
        is_international_remote=result["is_international_remote"],
        model_version=MODEL_VERSION,
        input_summary={
            "work_year":          body.work_year,
            "experience_level":   body.experience_level.value,
            "employment_type":    body.employment_type.value,
            "job_title_bucket":   body.job_title_bucket.value,
            "remote_ratio":       body.remote_ratio.value,
            "company_location":   body.company_location,
            "employee_residence": body.employee_residence,
            "company_size":       body.company_size.value,
        },
    )
