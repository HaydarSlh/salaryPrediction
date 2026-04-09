"""
schemas/prediction.py
---------------------
Pydantic models and enums for request/response validation.
"""

from pydantic import BaseModel, Field, model_validator
from enum import Enum


# ── Enums ─────────────────────────────────────────────────────────────────────

class ExperienceLevel(str, Enum):
    EN = "EN"   # Entry-level
    MI = "MI"   # Mid-level
    SE = "SE"   # Senior
    EX = "EX"   # Executive / Director


class EmploymentType(str, Enum):
    FT = "FT"   # Full-time
    PT = "PT"   # Part-time
    CT = "CT"   # Contract
    FL = "FL"   # Freelance


class JobTitleBucket(str, Enum):
    DATA_SCIENTIST       = "Data Scientist"
    DATA_ENGINEER        = "Data Engineer"
    DATA_ANALYST         = "Data Analyst"
    ML_ENGINEER          = "ML Engineer"
    RESEARCH_SPECIALIZED = "Research & Specialized"


class RemoteRatio(int, Enum):
    ON_SITE = 0    # Fully on-site
    HYBRID  = 50   # Hybrid
    REMOTE  = 100  # Fully remote


class CompanySize(str, Enum):
    SMALL  = "S"
    MEDIUM = "M"
    LARGE  = "L"


# ── Request ───────────────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    work_year:          int             = Field(..., ge=2020, le=2030,
                                                description="Year of the job (e.g. 2024)")
    experience_level:   ExperienceLevel = Field(...,
                                                description="EN | MI | SE | EX")
    employment_type:    EmploymentType  = Field(...,
                                                description="FT | PT | CT | FL")
    job_title_bucket:   JobTitleBucket  = Field(...,
                                                description="Broad role category")
    remote_ratio:       RemoteRatio     = Field(...,
                                                description="0 | 50 | 100")
    company_location:   str             = Field(..., min_length=2, max_length=2,
                                                description="ISO alpha-2 country code e.g. 'US'")
    employee_residence: str             = Field(..., min_length=2, max_length=2,
                                                description="ISO alpha-2 country code e.g. 'US'")
    company_size:       CompanySize     = Field(...,
                                                description="S | M | L")

    @model_validator(mode="before")
    @classmethod
    def normalise_country_codes(cls, values):
        """Force country codes to uppercase so 'us' and 'US' both work."""
        for field in ("company_location", "employee_residence"):
            if field in values and isinstance(values[field], str):
                values[field] = values[field].upper().strip()
        return values

    model_config = {
        "json_schema_extra": {
            "example": {
                "work_year":          2024,
                "experience_level":   "SE",
                "employment_type":    "FT",
                "job_title_bucket":   "Data Scientist",
                "remote_ratio":       100,
                "company_location":   "US",
                "employee_residence": "US",
                "company_size":       "M",
            }
        }
    }


# ── Response ──────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    predicted_salary_usd:    float = Field(..., description="Predicted annual salary in USD")
    is_international_remote: bool  = Field(..., description="True if employee lives abroad from company")
    model_version:           str   = Field(..., description="Model identifier")
    input_summary:           dict  = Field(..., description="Echo of the processed input")


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_version: str
