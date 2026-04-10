"""
supabase_service.py
-------------------
Handles all Supabase interactions.
Saves prediction results and LLM insights to their respective tables.

Schema:
    predictions  → one row per API call
    llm_insights → one row per Gemini response, linked to prediction
"""

import logging
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Lightweight Supabase REST client.
    Uses the Supabase REST API directly — no SDK required.
    """

    def __init__(self, url: str, api_key: str):
        self.base_url = url.rstrip("/")
        self.headers  = {
            "apikey":        api_key,
            "Authorization": f"Bearer {api_key}",
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        }

    def _post(self, table: str, data: dict) -> dict:
        """Insert a row and return the created record."""
        response = requests.post(
            f"{self.base_url}/rest/v1/{table}",
            headers=self.headers,
            json=data,
            timeout=15,
        )
        response.raise_for_status()
        return response.json()[0]

    # ── Public methods ────────────────────────────────────────────────────

    def save_prediction(self, prediction: dict) -> str:
        """
        Inserts a row into the predictions table.
        Returns the generated prediction ID.
        """
        row = {
            "work_year":               prediction["work_year"],
            "experience_level":        prediction["experience_level"],
            "employment_type":         prediction["employment_type"],
            "job_title_bucket":        prediction["job_title_bucket"],
            "remote_ratio":            prediction["remote_ratio"],
            "company_location":        prediction["company_location"],
            "employee_residence":      prediction["employee_residence"],
            "company_size":            prediction["company_size"],
            "is_international_remote": prediction["is_international_remote"],
            "predicted_salary_usd":    prediction["predicted_salary_usd"],
            "model_version":           prediction.get("model_version", "random-forest-v1.0"),
            "created_at":              datetime.now(timezone.utc).isoformat(),
        }

        result = self._post("predictions", row)
        prediction_id = result["id"]
        logger.info(f"Prediction saved — ID: {prediction_id}")
        return prediction_id

    def save_llm_insight(
        self,
        prediction_id: str,
        narrative: str,
        drivers: list,
        recommendations: list,
        chart_base64: str,
    ) -> str:
        """
        Inserts a row into the llm_insights table.
        Returns the generated insight ID.
        """
        import json

        row = {
            "prediction_id":   prediction_id,
            "narrative":       narrative,
            "drivers":         json.dumps(drivers),
            "recommendations": json.dumps(recommendations),
            "chart_base64":    chart_base64,
            "generated_at":    datetime.now(timezone.utc).isoformat(),
        }

        result = self._post("llm_insights", row)
        insight_id = result["id"]
        logger.info(f"LLM insight saved — ID: {insight_id}")
        return insight_id
