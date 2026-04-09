"""
config.py
---------
Central configuration for the Salary Prediction API.
All settings live here — change one place, affects the whole app.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).resolve().parent.parent
BUNDLE_PATH       = BASE_DIR / "models" / "salary_pipeline_bundle.joblib"

# ── Model metadata ────────────────────────────────────────────────────────────
MODEL_VERSION     = "random-forest-v1.0"
API_TITLE         = "Salary Prediction API"
API_DESCRIPTION   = (
    "Predicts annual salary (USD) for data science roles "
    "using a trained Random Forest model."
)
API_VERSION       = "1.0.0"
