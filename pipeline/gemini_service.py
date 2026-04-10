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
import requests

from market_stats import (
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
    "gemini-1.5-flash:generateContent"
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

    prompt = f"""
You are a senior data science career analyst. A user submitted their job profile and received a salary prediction. Your task is to generate a structured, insightful, and actionable salary analysis.

---

USER PROFILE:
- Work Year: {year}
- Job Title Category: {job}
- Experience Level: {exp_label}
- Employment Type: {emp_label}
- Company Location: {location}
- Company Size: {size_label}
- Remote Ratio: {remote_label}
- International Remote Worker: {"Yes" if is_intl else "No"}
- Predicted Salary: ${salary:,.0f} USD/year

---

MARKET STATISTICS (from real ds_salaries dataset):

Overall market:
- Mean salary: ${overall['mean']:,}
- Median salary: ${overall['median']:,}

Average salary by experience level:
- Entry:     ${exp_stats['Entry']:,}
- Mid:       ${exp_stats['Mid']:,}
- Senior:    ${exp_stats['Senior']:,}
- Executive: ${exp_stats['Executive']:,}

Average salary by job title:
- Data Analyst:           ${job_stats['Data Analyst']:,}
- Data Engineer:          ${job_stats['Data Engineer']:,}
- Data Scientist:         ${job_stats['Data Scientist']:,}
- ML Engineer:            ${job_stats['ML Engineer']:,}
- Research & Specialized: ${job_stats['Research & Specialized']:,}

Average salary by company size:
- Small:  ${size_stats['Small']:,}
- Medium: ${size_stats['Medium']:,}
- Large:  ${size_stats['Large']:,}

Average salary by remote ratio:
- On-site: ${rem_stats['On-site']:,}
- Hybrid:  ${rem_stats['Hybrid']:,}
- Remote:  ${rem_stats['Remote']:,}

---

INSTRUCTIONS:
Generate a response as a valid JSON object with EXACTLY these 4 keys:

1. "narrative": A 3-4 sentence paragraph placing this salary in market context. 
   Be specific — mention the exact salary, how it compares to the market average 
   for this role and experience level, and what that means for the user.

2. "drivers": A list of 3 strings. Each string explains one factor from their 
   profile that is significantly helping or hurting their salary. Be specific 
   with numbers from the market stats above.

3. "recommendations": A list of 4 objects, each with:
   - "action": short title of the recommendation
   - "impact": estimated salary increase in USD (number only, e.g. 25000)
   - "explanation": one sentence explaining why this works, with specific numbers

   Rank recommendations from highest to lowest impact.
   Base the impact estimates on the differences in the market stats above.
   Focus on realistic, actionable steps the employee can take.

4. "chart_data": An object with:
   - "title": "Your Salary vs Market Average by Experience Level"
   - "labels": ["Entry", "Mid", "Senior", "Executive"]
   - "market_values": [{exp_stats['Entry']}, {exp_stats['Mid']}, {exp_stats['Senior']}, {exp_stats['Executive']}]
   - "user_experience": "{exp_label}"
   - "user_salary": {salary:.0f}

Return ONLY the JSON object. No markdown, no backticks, no explanation outside the JSON.
"""
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
            "temperature": 0.4,       # lower = more factual, less creative
            "maxOutputTokens": 1500,
        },
    }

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

    except requests.exceptions.Timeout:
        logger.error("Gemini API request timed out.")
        raise RuntimeError("Gemini API timed out. Try again.")

    except requests.exceptions.HTTPError as e:
        logger.error(f"Gemini API HTTP error: {e} | {response.text}")
        raise RuntimeError(f"Gemini API error: {response.status_code}")

    # ── Parse response ────────────────────────────────────────────────────
    try:
        raw_text = (
            response.json()
            ["candidates"][0]
            ["content"]
            ["parts"][0]
            ["text"]
        )

        # Strip markdown fences if Gemini adds them despite instructions
        clean_text = raw_text.strip()
        if clean_text.startswith("```"):
            clean_text = clean_text.split("```")[1]
            if clean_text.startswith("json"):
                clean_text = clean_text[4:]

        parsed = json.loads(clean_text.strip())

    except (KeyError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse Gemini response: {e}\nRaw: {raw_text}")
        raise RuntimeError("Gemini returned an unparseable response.")

    logger.info("Gemini response parsed successfully.")
    return parsed
