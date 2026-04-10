"""
pipeline.py
-----------
Main orchestrator for the local pipeline.

Flow:
    1. Call FastAPI /predict with job details
    2. Call Gemini with prediction + market context
    3. Build chart from Gemini chart_data
    4. Save prediction + insights to Supabase
"""

import os
import logging
import requests
from dotenv import load_dotenv

from gemini_service import call_gemini
from chart_builder import build_salary_chart
from supabase_service import SupabaseClient

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load env vars ─────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

FASTAPI_URL   = os.getenv("FASTAPI_URL",   "http://127.0.0.1:8000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL  = os.getenv("SUPABASE_URL")
SUPABASE_KEY  = os.getenv("SUPABASE_KEY")


# ── Step 1: Call FastAPI ──────────────────────────────────────────────────────

def call_fastapi(job_input: dict) -> dict:
    """Sends job details to FastAPI and returns the prediction response."""
    logger.info(f"Calling FastAPI at {FASTAPI_URL}/predict ...")
    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict",
            json=job_input,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(f"Prediction received: ${result['predicted_salary_usd']:,.0f}")
        return result

    except requests.exceptions.ConnectionError:
        logger.error("Cannot connect to FastAPI. Is it running?")
        raise
    except requests.exceptions.HTTPError as e:
        logger.error(f"FastAPI error: {e} | {response.text}")
        raise


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(job_input: dict):
    """
    Runs the full pipeline for one job profile.

    Parameters
    ----------
    job_input : dict
        Raw job details — same fields as the FastAPI request body.
    """

    logger.info("=" * 60)
    logger.info("Starting pipeline run ...")

    # ── 1. Get salary prediction ──────────────────────────────────────────
    prediction_response = call_fastapi(job_input)

    # Merge original input into prediction for Gemini context
    full_prediction = {
        **job_input,
        "predicted_salary_usd":    prediction_response["predicted_salary_usd"],
        "is_international_remote": prediction_response["is_international_remote"],
        "model_version":           prediction_response["model_version"],
    }

    # ── 2. Call Gemini ────────────────────────────────────────────────────
    logger.info("Calling Gemini for insights ...")
    gemini_result = call_gemini(full_prediction, api_key=GEMINI_API_KEY)

    narrative       = gemini_result["narrative"]
    drivers         = gemini_result["drivers"]
    recommendations = gemini_result["recommendations"]
    chart_data      = gemini_result["chart_data"]

    logger.info(f"Narrative: {narrative[:80]}...")
    logger.info(f"Recommendations: {len(recommendations)} returned")

    # ── 3. Build chart ────────────────────────────────────────────────────
    logger.info("Building chart ...")
    chart_base64 = build_salary_chart(chart_data)

    # ── 4. Save to Supabase ───────────────────────────────────────────────
    logger.info("Saving to Supabase ...")
    db = SupabaseClient(url=SUPABASE_URL, api_key=SUPABASE_KEY)

    prediction_id = db.save_prediction(full_prediction)
    insight_id    = db.save_llm_insight(
        prediction_id=prediction_id,
        narrative=narrative,
        drivers=drivers,
        recommendations=recommendations,
        chart_base64=chart_base64,
    )

    logger.info("=" * 60)
    logger.info(f"✅ Pipeline complete.")
    logger.info(f"   Prediction ID : {prediction_id}")
    logger.info(f"   Insight ID    : {insight_id}")
    logger.info(f"   Salary        : ${full_prediction['predicted_salary_usd']:,.0f}")
    logger.info("=" * 60)

    return {
        "prediction_id":        prediction_id,
        "insight_id":           insight_id,
        "predicted_salary_usd": full_prediction["predicted_salary_usd"],
        "narrative":            narrative,
        "recommendations":      recommendations,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example job profile — change these values to test different scenarios
    sample_input = {
        "work_year":          2024,
        "experience_level":   "SE",
        "employment_type":    "FT",
        "job_title_bucket":   "Data Scientist",
        "remote_ratio":       100,
        "company_location":   "US",
        "employee_residence": "US",
        "company_size":       "M",
    }

    result = run_pipeline(sample_input)
