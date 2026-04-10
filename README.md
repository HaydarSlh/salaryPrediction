# Salary Prediction API

A full-stack salary prediction system for data different roles. Users submit a job profile, get an ML-predicted salary, and receive AI-generated market insights — all persisted to a database and visualized in a dashboard.

---

## Live Deployments

| Service | URL |
|---|---|
| Streamlit Dashboard | https://salaryprediction-9kyatd7zjvrh98icngrfvm.streamlit.app/ |
| FastAPI (Railway) | https://salaryprediction-production-d26f.up.railway.app |
| FastAPI Health Check | https://salaryprediction-production-d26f.up.railway.app/health |
| FastAPI Docs | https://salaryprediction-production-d26f.up.railway.app/docs |

---

## Architecture

```
User Input (Streamlit)
       │
       ▼
POST /predict  ──→  Random Forest Model  ──→  Predicted Salary (USD)
       │
       ▼
Gemini 2.5 Flash  ──→  narrative + drivers + recommendations
       │
       ▼
Matplotlib  ──→  Salary vs Market Chart (base64 PNG)
       │
       ▼
Supabase  ──→  predictions + llm_insights tables
       │
       ▼
Streamlit Dashboard  ──→  Results + History
```

---

## Project Structure

```
salary_api/
│
├── main.py                        # FastAPI app entry point
├── Dockerfile                     # Railway deployment config
├── requirements.txt               # All dependencies
├── streamlit_app.py               # Streamlit dashboard
│
├── app/
│   ├── config.py                  # Paths, model metadata, API info
│   ├── routers/
│   │   ├── health.py              # GET /health
│   │   └── predict.py             # POST /predict
│   ├── schemas/
│   │   └── prediction.py          # Request/response Pydantic models
│   ├── services/
│   │   └── predictor.py           # ML prediction logic
│   ├── models/                    # Trained model + encoders (.pkl, .joblib)
│   └── data/
│       └── ds_salaries.csv        # Training dataset (608 rows, 2020–2024)
│
└── pipeline/
    ├── pipeline.py                # Orchestrates all 4 steps
    ├── gemini_service.py          # Gemini API calls + retry logic
    ├── chart_builder.py           # Matplotlib bar chart → base64 PNG
    ├── supabase_service.py        # Supabase REST API client
    └── market_stats.py            # Pre-computed market benchmarks
```

---

## API Reference

### `GET /health`

Returns model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "random-forest-v1.0"
}
```

### `POST /predict`

**Request body:**

```json
{
  "work_year": 2024,
  "experience_level": "SE",
  "employment_type": "FT",
  "job_title_bucket": "Data Scientist",
  "remote_ratio": 100,
  "company_location": "US",
  "employee_residence": "US",
  "company_size": "M"
}
```

| Field | Type | Values |
|---|---|---|
| `work_year` | int | 2020–2030 |
| `experience_level` | enum | `EN` Entry / `MI` Mid / `SE` Senior / `EX` Executive |
| `employment_type` | enum | `FT` Full-time / `PT` Part-time / `CT` Contract / `FL` Freelance |
| `job_title_bucket` | enum | `Data Scientist` / `Data Engineer` / `Data Analyst` / `ML Engineer` / `Research & Specialized` |
| `remote_ratio` | enum | `0` On-site / `50` Hybrid / `100` Remote |
| `company_location` | string | ISO 3166-1 alpha-2 country code (e.g. `US`, `GB`) |
| `employee_residence` | string | ISO 3166-1 alpha-2 country code |
| `company_size` | enum | `S` Small / `M` Medium / `L` Large |

**Response:**

```json
{
  "predicted_salary_usd": 160804.0,
  "is_international_remote": false,
  "model_version": "random-forest-v1.0",
  "input_summary": { ... }
}
```

---

## ML Model

- **Algorithm**: Random Forest Regressor
- **Target**: Log-transformed salary (inverse-transformed on output via `expm1`)
- **Dataset**: `ds_salaries.csv` — 608 rows, 2020–2024
- **Encoding strategy**:
  - Ordinal: experience level, company size
  - Label encoding: employment type, job title
  - Target encoding: company location (mean salary per country, global mean fallback)

**Market benchmarks (from training data):**

| Experience | Avg Salary |
|---|---|
| Entry | $61,643 |
| Mid | $87,468 |
| Senior | $138,582 |
| Executive | $191,354 |

---

## Pipeline

The local pipeline (`pipeline/pipeline.py`) runs 4 steps in sequence:

1. **Call FastAPI** — sends job profile, gets predicted salary
2. **Call Gemini** — sends prediction + market context, gets structured JSON insights
3. **Build chart** — generates grouped bar chart (market avg vs user salary) as base64 PNG
4. **Save to Supabase** — inserts into `predictions` and `llm_insights` tables

### Gemini Output Format

```json
{
  "narrative": "2-3 sentence market comparison",
  "drivers": ["factor 1", "factor 2", "factor 3"],
  "recommendations": [
    { "action": "Title", "impact": 20000, "explanation": "One sentence." }
  ]
}
```

### Gemini Retry Logic

- Model: `gemini-2.5-flash`
- Max retries: 5
- Backoff: 5s → 10s → 20s → 40s → 80s
- Handles: 429 (quota), 503 (overload), timeout
- Daily quota exhaustion detected and reported immediately

---

## Database (Supabase)

**Table: `predictions`**

| Column | Type | Description |
|---|---|---|
| `id` | uuid | Auto-generated |
| `work_year` | int | Input field |
| `experience_level` | text | EN/MI/SE/EX |
| `employment_type` | text | FT/PT/CT/FL |
| `job_title_bucket` | text | Job category |
| `remote_ratio` | int | 0/50/100 |
| `company_location` | text | Country code |
| `employee_residence` | text | Country code |
| `company_size` | text | S/M/L |
| `is_international_remote` | bool | Derived feature |
| `predicted_salary_usd` | float | Model output |
| `model_version` | text | e.g. random-forest-v1.0 |
| `created_at` | timestamp | Auto-generated |

**Table: `llm_insights`**

| Column | Type | Description |
|---|---|---|
| `id` | uuid | Auto-generated |
| `prediction_id` | uuid | FK → predictions.id |
| `narrative` | text | Gemini narrative |
| `drivers` | text | JSON array of strings |
| `recommendations` | text | JSON array of objects |
| `chart_base64` | text | Base64-encoded PNG |
| `generated_at` | timestamp | Auto-generated |

---

## Environment Variables

Create a `.env` file in the project root:

```env
FASTAPI_URL=https://salaryprediction-production-d26f.up.railway.app
GEMINI_API_KEY=your_gemini_api_key
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key
```

> `.env` is listed in `.gitignore` and must never be committed.

---

## Local Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Set up `.env`** (see above)

**3. Run the Streamlit dashboard**

```bash
streamlit run streamlit_app.py
```

**4. Run the pipeline directly (for testing)**

```bash
cd pipeline
py pipeline.py
```

---

## Deployment (Railway)

The FastAPI service is deployed on Railway using the included `Dockerfile`.

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .
COPY app/ app/
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Railway auto-deploys on push to `main`. The `/health` endpoint is used for liveness checks.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.115.0 | REST API framework |
| uvicorn | 0.30.6 | ASGI server |
| pydantic | 2.9.2 | Input validation |
| scikit-learn | 1.5.2 | ML model + encoders |
| joblib | 1.4.2 | Model serialization |
| streamlit | 1.40.0 | Dashboard UI |
| requests | 2.32.3 | HTTP calls (Gemini, Supabase, FastAPI) |
| matplotlib | 3.9.2 | Chart generation |
| numpy | 1.26.4 | Numerical operations |
| pandas | 2.2.3 | Data handling |
| python-dotenv | 1.0.1 | Environment variable loading |
