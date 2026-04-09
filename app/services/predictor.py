"""
services/predictor.py
---------------------
SalaryPredictor loads the pipeline bundle once at startup
and exposes a single predict() method.
Completely decoupled from FastAPI — testable in isolation.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from app.config import BUNDLE_PATH, MODEL_VERSION

logger = logging.getLogger(__name__)

# Feature order must match the column order used during training
FEATURE_COLUMNS = [
    "work_year",
    "experience_level",
    "employment_type",
    "job_title_bucket",
    "remote_ratio",
    "company_location",
    "company_size",
    "is_international_remote",
]


class SalaryPredictor:
    """
    Wraps the pipeline bundle (model + all encoders).
    One instance is created at app startup and reused for every request.
    """

    def __init__(self):
        if not Path(BUNDLE_PATH).exists():
            raise FileNotFoundError(
                f"Pipeline bundle not found at '{BUNDLE_PATH}'.\n"
                "Run the training notebook and copy "
                "'salary_pipeline_bundle.joblib' into the models/ folder."
            )

        logger.info(f"Loading pipeline bundle from {BUNDLE_PATH} ...")
        bundle = joblib.load(BUNDLE_PATH)

        # ── Unpack every component from the bundle ────────────────────────────
        self.model       = bundle["model"]
        self.le_emp      = bundle["le_employment_type"]
        self.le_job      = bundle["le_job_title_bucket"]
        self.exp_ord     = bundle["ordinal_exp_level"]
        self.size_ord    = bundle["ordinal_company_size"]
        self.loc_means   = bundle["target_enc_location"]
        self.global_mean = bundle["target_enc_global"]

        logger.info("Pipeline bundle loaded successfully.")

    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        """
        Parameters
        ----------
        data : dict
            Already-validated input from PredictionRequest.model_dump().

        Returns
        -------
        dict
            predicted_salary_usd, is_international_remote, features_used
        """

        # ── Engineer is_international_remote ─────────────────────────────────
        is_intl_remote = int(
            data["company_location"] != data["employee_residence"]
        )

        # ── Ordinal encoding ──────────────────────────────────────────────────
        try:
            exp_encoded = self.exp_ord[data["experience_level"]]
        except KeyError:
            raise ValueError(f"Unknown experience_level: '{data['experience_level']}'")

        try:
            size_encoded = self.size_ord[data["company_size"]]
        except KeyError:
            raise ValueError(f"Unknown company_size: '{data['company_size']}'")

        # ── Label encoding ────────────────────────────────────────────────────
        try:
            emp_encoded = int(self.le_emp.transform([data["employment_type"]])[0])
        except Exception:
            raise ValueError(f"Unknown employment_type: '{data['employment_type']}'")

        try:
            job_encoded = int(self.le_job.transform([data["job_title_bucket"]])[0])
        except Exception:
            raise ValueError(f"Unknown job_title_bucket: '{data['job_title_bucket']}'")

        # ── Target encoding (unseen country → global mean fallback) ───────────
        loc_encoded = float(
            self.loc_means.get(data["company_location"], self.global_mean)
        )

        # ── Assemble feature row ──────────────────────────────────────────────
        features = {
            "work_year":               data["work_year"],
            "experience_level":        exp_encoded,
            "employment_type":         emp_encoded,
            "job_title_bucket":        job_encoded,
            "remote_ratio":            data["remote_ratio"],
            "company_location":        loc_encoded,
            "company_size":            size_encoded,
            "is_international_remote": is_intl_remote,
        }

        X = pd.DataFrame([features], columns=FEATURE_COLUMNS)

        # ── Predict → inverse log transform → USD ────────────────────────────
        log_pred   = self.model.predict(X)[0]
        salary_usd = float(np.expm1(log_pred))

        return {
            "predicted_salary_usd":    round(salary_usd, 2),
            "is_international_remote": bool(is_intl_remote),
            "features_used":           features,
        }

    @property
    def version(self) -> str:
        return MODEL_VERSION
