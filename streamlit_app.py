"""
streamlit_app.py
----------------
Salary Prediction Dashboard.
Reads exclusively from Supabase.
Flow: User fills form → pipeline runs → Supabase updated → results displayed.
"""

import os
import json
import base64
import requests
import streamlit as st
from datetime import datetime

FASTAPI_URL  = os.getenv("FASTAPI_URL",  "http://127.0.0.1:8000")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_KEY   = os.getenv("GEMINI_API_KEY")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💼",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .salary-box {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 16px;
        padding: 32px;
        text-align: center;
        margin-bottom: 24px;
    }
    .salary-label {
        color: #a0c4e8;
        font-size: 16px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .salary-value {
        color: #ffffff;
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -1px;
    }
    .salary-sub {
        color: #7fb3d3;
        font-size: 13px;
        margin-top: 6px;
    }
    .narrative-box {
        background: #1a1f2e;
        border-left: 4px solid #4C72B0;
        border-radius: 8px;
        padding: 20px 24px;
        margin-bottom: 20px;
        color: #d0d8e8;
        font-size: 15px;
        line-height: 1.7;
    }
    .driver-card {
        background: #1a1f2e;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        color: #c8d4e8;
        font-size: 14px;
        border: 1px solid #2a3040;
    }
    .rec-card {
        background: #1a2e1a;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border: 1px solid #2a4a2a;
    }
    .rec-action {
        color: #7ecf7e;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .rec-impact {
        color: #a8e6a8;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .rec-explanation {
        color: #b0c8b0;
        font-size: 13px;
        line-height: 1.5;
    }
    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #e0e8f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #2a3040;
    }
    .history-badge {
        background: #2a3040;
        border-radius: 6px;
        padding: 2px 10px;
        font-size: 12px;
        color: #a0b0c8;
    }
    div[data-testid="stSidebar"] {
        background: #0f1420;
    }
</style>
""", unsafe_allow_html=True)


# ── Supabase helpers ──────────────────────────────────────────────────────────

def supabase_headers():
    return {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type":  "application/json",
    }


def fetch_latest_prediction():
    """Fetch the most recent prediction + its insight from Supabase."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/predictions",
            headers=supabase_headers(),
            params={
                "select": "*, llm_insights(*)",
                "order":  "created_at.desc",
                "limit":  "1",
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data[0] if data else None
    except Exception as e:
        st.error(f"Failed to fetch from Supabase: {e}")
        return None


def fetch_prediction_history():
    """Fetch last 10 predictions for the history table."""
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/predictions",
            headers=supabase_headers(),
            params={
                "select": "id,created_at,experience_level,job_title_bucket,company_location,company_size,predicted_salary_usd",
                "order":  "created_at.desc",
                "limit":  "10",
            },
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Failed to fetch history: {e}")
        return []


# ── Pipeline trigger ──────────────────────────────────────────────────────────

def run_pipeline(job_input: dict):
    """
    Calls FastAPI → Gemini → Supabase.
    Imports pipeline functions directly.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from pipeline.gemini_service import call_gemini
    from pipeline.chart_builder import build_salary_chart
    from pipeline.supabase_service import SupabaseClient

    # 1. FastAPI
    r = requests.post(f"{FASTAPI_URL}/predict", json=job_input, timeout=15)
    r.raise_for_status()
    api_result = r.json()

    full_prediction = {
        **job_input,
        "predicted_salary_usd":    api_result["predicted_salary_usd"],
        "is_international_remote": api_result["is_international_remote"],
        "model_version":           api_result["model_version"],
    }

    # 2. Gemini
    gemini_result = call_gemini(full_prediction, api_key=GEMINI_KEY)

    # 3. Chart
    chart_base64 = build_salary_chart(full_prediction)

    # 4. Supabase
    db = SupabaseClient(url=SUPABASE_URL, api_key=SUPABASE_KEY)
    prediction_id = db.save_prediction(full_prediction)
    db.save_llm_insight(
        prediction_id=prediction_id,
        narrative=gemini_result["narrative"],
        drivers=gemini_result["drivers"],
        recommendations=gemini_result["recommendations"],
        chart_base64=chart_base64,
    )
    return prediction_id


# ── Display helpers ───────────────────────────────────────────────────────────

def display_salary(salary: float, is_intl: bool):
    intl_note = "🌍 International Remote" if is_intl else "🏢 Local Employee"
    st.markdown(f"""
    <div class="salary-box">
        <div class="salary-label">Predicted Annual Salary</div>
        <div class="salary-value">${salary:,.0f}</div>
        <div class="salary-sub">{intl_note} &nbsp;|&nbsp; USD / year</div>
    </div>
    """, unsafe_allow_html=True)


def display_narrative(narrative: str):
    st.markdown('<div class="section-title">📊 Market Analysis</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="narrative-box">{narrative}</div>', unsafe_allow_html=True)


def display_drivers(drivers):
    if isinstance(drivers, str):
        drivers = json.loads(drivers)
    st.markdown('<div class="section-title">⚡ Salary Drivers</div>', unsafe_allow_html=True)
    icons = ["🟢", "🔵", "🟡"]
    for i, driver in enumerate(drivers):
        icon = icons[i] if i < len(icons) else "•"
        st.markdown(f'<div class="driver-card">{icon} &nbsp;{driver}</div>', unsafe_allow_html=True)


def display_recommendations(recommendations):
    if isinstance(recommendations, str):
        recommendations = json.loads(recommendations)
    st.markdown('<div class="section-title">🚀 How to Increase Your Salary</div>', unsafe_allow_html=True)
    for rec in recommendations:
        impact = rec.get("impact", 0)
        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-action">💡 {rec.get('action', '')}</div>
            <div class="rec-impact">+${impact:,} estimated increase</div>
            <div class="rec-explanation">{rec.get('explanation', '')}</div>
        </div>
        """, unsafe_allow_html=True)


def display_chart(chart_base64: str):
    st.markdown('<div class="section-title">📈 Your Salary vs Market</div>', unsafe_allow_html=True)
    try:
        img_bytes = base64.b64decode(chart_base64)
        st.image(img_bytes, use_container_width=True)
    except Exception:
        st.warning("Chart could not be displayed.")


def display_history(records):
    st.markdown('<div class="section-title">🕓 Prediction History</div>', unsafe_allow_html=True)
    if not records:
        st.info("No predictions yet.")
        return

    EXP = {"EN": "Entry", "MI": "Mid", "SE": "Senior", "EX": "Executive"}
    SIZE = {"S": "Small", "M": "Medium", "L": "Large"}

    rows = []
    for r in records:
        rows.append({
            "Date":          datetime.fromisoformat(r["created_at"].replace("Z", "")).strftime("%Y-%m-%d %H:%M"),
            "Role":          r["job_title_bucket"],
            "Experience":    EXP.get(r["experience_level"], r["experience_level"]),
            "Location":      r["company_location"],
            "Company Size":  SIZE.get(r["company_size"], r["company_size"]),
            "Predicted ($)": f"${r['predicted_salary_usd']:,.0f}",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)


# ── Sidebar — input form ──────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 💼 Job Profile")
    st.markdown("Fill in your details to get a salary prediction and personalized insights.")
    st.markdown("---")

    work_year = st.selectbox("Work Year", [2024, 2023, 2022, 2021, 2020])

    experience_level = st.selectbox(
        "Experience Level",
        ["EN", "MI", "SE", "EX"],
        format_func=lambda x: {"EN": "Entry Level", "MI": "Mid Level",
                                "SE": "Senior", "EX": "Executive"}[x]
    )

    employment_type = st.selectbox(
        "Employment Type",
        ["FT", "PT", "CT", "FL"],
        format_func=lambda x: {"FT": "Full-Time", "PT": "Part-Time",
                                "CT": "Contract", "FL": "Freelance"}[x]
    )

    job_title_bucket = st.selectbox(
        "Job Category",
        ["Data Scientist", "Data Engineer", "Data Analyst",
         "ML Engineer", "Research & Specialized"]
    )

    remote_ratio = st.selectbox(
        "Work Mode",
        [0, 50, 100],
        format_func=lambda x: {0: "On-site", 50: "Hybrid", 100: "Fully Remote"}[x]
    )

    company_location = st.text_input(
        "Company Location (country code)",
        value="US",
        max_chars=2,
    ).upper()

    employee_residence = st.text_input(
        "Your Country (country code)",
        value="US",
        max_chars=2,
    ).upper()

    company_size = st.selectbox(
        "Company Size",
        ["S", "M", "L"],
        format_func=lambda x: {"S": "Small", "M": "Medium", "L": "Large"}[x]
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Predict My Salary", use_container_width=True, type="primary")


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("# 💼 Data Science Salary Predictor")
st.markdown("AI-powered salary prediction with personalized career insights.")
st.markdown("---")

# ── Handle prediction ─────────────────────────────────────────────────────────
if predict_btn:
    job_input = {
        "work_year":          work_year,
        "experience_level":   experience_level,
        "employment_type":    employment_type,
        "job_title_bucket":   job_title_bucket,
        "remote_ratio":       remote_ratio,
        "company_location":   company_location,
        "employee_residence": employee_residence,
        "company_size":       company_size,
    }

    with st.spinner("Running prediction pipeline — this takes 10–20 seconds..."):
        try:
            run_pipeline(job_input)
            st.success("✅ Prediction complete!")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

# ── Load and display latest result ────────────────────────────────────────────
latest = fetch_latest_prediction()

if latest is None:
    st.info("👈 Fill in your job profile on the left and click **Predict My Salary** to get started.")

else:
    insight = latest.get("llm_insights", [{}])
    insight = insight[0] if isinstance(insight, list) and insight else {}

    # ── Row 1: salary + narrative ─────────────────────────────────────────
    col1, col2 = st.columns([1, 2])

    with col1:
        display_salary(
            salary=latest["predicted_salary_usd"],
            is_intl=latest["is_international_remote"],
        )

        if insight.get("drivers"):
            display_drivers(insight["drivers"])

    with col2:
        if insight.get("narrative"):
            display_narrative(insight["narrative"])

        if insight.get("recommendations"):
            display_recommendations(insight["recommendations"])

    # ── Row 2: chart ──────────────────────────────────────────────────────
    if insight.get("chart_base64"):
        display_chart(insight["chart_base64"])

    st.markdown("---")

    # ── Row 3: history ────────────────────────────────────────────────────
    history = fetch_prediction_history()
    display_history(history)
