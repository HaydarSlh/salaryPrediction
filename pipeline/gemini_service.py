"""
gemini_service.py
-----------------
Calls the Gemini API with the prediction result + market context.
Returns structured JSON containing:
  - narrative:        market position paragraph
  - drivers:          what's helping/hurting the salary
  - recommendations:  ranked actions to increase salary
  - chart_data:       numbers for the salary-by-experience bar chart
"""

import os
import json
import logging
import time
import requests

from pipeline.market_stats import (
    MARKET_STATS,
    EXPERIENCE_LABELS,
    COMPANY_SIZE_LABELS,
    REMOTE_LABELS,
    EMPLOYMENT_LABELS,
    EXPERIENCE_ORDER,
)

logger = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)


def _build_prompt(prediction: dict) -> str:
    """
    Constructs the full prompt sent to Gemini.
    Includes prediction result + pre-computed market stats as context.
    """

    # ── Human-readable labels ─────────────────────────────────────────────
    exp_label    = EXPERIENCE_LABELS[prediction["experience_level"]]
    size_label   = COMPANY_SIZE_LABELS[prediction["company_size"]]
    remote_label = REMOTE_LABELS[prediction["remote_ratio"]]
    emp_label    = EMPLOYMENT_LABELS[prediction["employment_type"]]
    salary       = prediction["predicted_salary_usd"]
    job          = prediction["job_title_bucket"]
    location     = prediction["company_location"]
    year         = prediction["work_year"]
    is_intl      = prediction["is_international_remote"]

    # ── Market context string ─────────────────────────────────────────────
    exp_stats  = MARKET_STATS["by_experience"]
    job_stats  = MARKET_STATS["by_job_title"]
    size_stats = MARKET_STATS["by_company_size"]
    rem_stats  = MARKET_STATS["by_remote_ratio"]
    overall    = MARKET_STATS["overall"]

    prompt = f"""You are a data science career analyst. Given the profile and market stats below, return a JSON object with exactly 3 keys: "narrative", "drivers", "recommendations".

PROFILE: {exp_label} {job} at a {size_label} company in {location}, {remote_label}, {emp_label}. Predicted salary: ${salary:,.0f}/yr.

MARKET STATS:
- Overall mean/median: ${overall['mean']:,} / ${overall['median']:,}
- By experience: Entry ${exp_stats['Entry']:,}, Mid ${exp_stats['Mid']:,}, Senior ${exp_stats['Senior']:,}, Executive ${exp_stats['Executive']:,}
- By job: Analyst ${job_stats['Data Analyst']:,}, Engineer ${job_stats['Data Engineer']:,}, Scientist ${job_stats['Data Scientist']:,}, ML Eng ${job_stats['ML Engineer']:,}
- By size: Small ${size_stats['Small']:,}, Medium ${size_stats['Medium']:,}, Large ${size_stats['Large']:,}
- By remote: On-site ${rem_stats['On-site']:,}, Hybrid ${rem_stats['Hybrid']:,}, Remote ${rem_stats['Remote']:,}

OUTPUT FORMAT:
{{"narrative": "2-3 sentences comparing salary to market","drivers": ["factor 1","factor 2","factor 3"],"recommendations": [{{"action": "title","impact": 20000,"explanation": "one sentence"}}]}}

Return only the JSON. No markdown."""
    return prompt.strip()


def call_gemini(prediction: dict, api_key: str) -> dict:
    """
    Calls Gemini API and returns parsed structured response.

    Parameters
    ----------
    prediction : dict
        The full prediction result from FastAPI + original input fields.
    api_key : str
        Gemini API key.

    Returns
    -------
    dict with keys: narrative, drivers, recommendations, chart_data
    """
    prompt = _build_prompt(prediction)

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.4,
            "responseMimeType": "application/json",
        },
    }

    max_retries = 5
    backoff = 5  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            break  # success

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(f"Gemini timeout (attempt {attempt}/{max_retries}), retrying in {wait}s ...")
                time.sleep(wait)
                continue
            logger.error("Gemini API request timed out after all retries.")
            raise RuntimeError("Gemini API timed out. Try again.")

        except requests.exceptions.HTTPError as e:
            status = response.status_code
            if status in (429, 503) and attempt < max_retries:
                # Use retry delay from response if available, else exponential backoff
                try:
                    retry_delay = (
                        response.json()
                        .get("error", {})
                        .get("details", [{}])[-1]
                        .get("retryDelay", "")
                    )
                    wait = int(retry_delay.rstrip("s")) if retry_delay else backoff * (2 ** (attempt - 1))
                except Exception:
                    wait = backoff * (2 ** (attempt - 1))

                # Daily quota exhaustion — no point retrying
                if status == 429:
                    err_body = response.json().get("error", {})
                    for violation in err_body.get("details", []):
                        for v in violation.get("violations", []):
                            if "PerDay" in v.get("quotaId", ""):
                                logger.error("Gemini daily quota exhausted. Wait until midnight PT or use a new project.")
                                raise RuntimeError("Gemini daily quota exhausted.")

                logger.warning(f"Gemini {status} (attempt {attempt}/{max_retries}), retrying in {wait}s ...")
                time.sleep(wait)
            else:
                safe_url = str(e).replace(api_key, "***")
                logger.error(f"Gemini API HTTP error: {safe_url} | {response.text}")
                raise RuntimeError(f"Gemini API error: {status}")

    # ── Parse response ────────────────────────────────────────────────────
    try:
        raw_text = (
            response.json()
            ["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
        )

        parsed = json.loads(raw_text.strip())

    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse Gemini response: {e}\nRaw: {raw_text}")
        raise RuntimeError("Gemini returned an unparseable response.")

    # Build chart_data locally — no need to ask Gemini for static numbers
    exp_stats = MARKET_STATS["by_experience"]
    exp_label = EXPERIENCE_LABELS[prediction["experience_level"]]
    parsed["chart_data"] = {
        "title": "Your Salary vs Market Average by Experience Level",
        "labels": ["Entry", "Mid", "Senior", "Executive"],
        "market_values": [exp_stats["Entry"], exp_stats["Mid"], exp_stats["Senior"], exp_stats["Executive"]],
        "user_experience": exp_label,
        "user_salary": int(prediction["predicted_salary_usd"]),
    }

    logger.info("Gemini response parsed successfully.")
    return parsed
