"""
market_stats.py
---------------
Pre-computed market statistics from the ds_salaries dataset.
These are hardcoded so the pipeline doesn't need the CSV at runtime.
All salary values are in USD.
"""

MARKET_STATS = {
    "overall": {
        "mean":   111752,
        "median": 101570,
        "min":    2859,
        "max":    403500,
    },

    "by_experience": {
        "Entry":     61643,
        "Mid":       87468,
        "Senior":    138582,
        "Executive": 191354,
    },

    "by_job_title": {
        "Data Analyst":          97148,
        "Data Engineer":         114109,
        "Data Scientist":        119016,
        "ML Engineer":           107453,
        "Research & Specialized": 120628,
    },

    "by_company_size": {
        "Small":  77482,
        "Medium": 116763,
        "Large":  117867,
    },

    "by_remote_ratio": {
        "On-site": 105988,
        "Hybrid":  80626,
        "Remote":  121761,
    },

    "by_employment_type": {
        "Full-Time":  112926,
        "Part-Time":  33070,
        "Contract":   182075,
        "Freelance":  48000,
    },
}

# Human-readable label maps — same as used in training
EXPERIENCE_LABELS = {
    "EN": "Entry",
    "MI": "Mid",
    "SE": "Senior",
    "EX": "Executive",
}

EXPERIENCE_ORDER = ["Entry", "Mid", "Senior", "Executive"]

COMPANY_SIZE_LABELS = {
    "S": "Small",
    "M": "Medium",
    "L": "Large",
}

REMOTE_LABELS = {
    0:   "On-site",
    50:  "Hybrid",
    100: "Remote",
}

EMPLOYMENT_LABELS = {
    "FT": "Full-Time",
    "PT": "Part-Time",
    "CT": "Contract",
    "FL": "Freelance",
}
