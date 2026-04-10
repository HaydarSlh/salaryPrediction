"""
chart_builder.py
----------------
Builds salary comparison chart directly from market stats.
No longer depends on Gemini returning chart_data.
"""

import io
import base64
import logging
import matplotlib.pyplot as plt
import numpy as np

from market_stats import (
    MARKET_STATS,
    EXPERIENCE_LABELS,
    EXPERIENCE_ORDER,
)

logger = logging.getLogger(__name__)

COLOR_MARKET = "#4C72B0"
COLOR_USER   = "#DD8452"


def build_salary_chart(prediction: dict) -> str:
    """
    Builds a bar chart comparing market average vs user predicted salary
    by experience level.

    Parameters
    ----------
    prediction : dict
        Full prediction dict containing experience_level and predicted_salary_usd.

    Returns
    -------
    str — base64-encoded PNG
    """
    exp_label  = EXPERIENCE_LABELS[prediction["experience_level"]]
    user_salary = prediction["predicted_salary_usd"]
    labels      = EXPERIENCE_ORDER
    market_vals = [MARKET_STATS["by_experience"][l] for l in labels]

    x     = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # Market average bars
    market_bars = ax.bar(
        x - width / 2, market_vals, width,
        color=COLOR_MARKET, alpha=0.85, label="Market Average"
    )

    # User salary — only on their experience level
    user_vals = [0] * len(labels)
    if exp_label in labels:
        user_vals[labels.index(exp_label)] = user_salary

    user_bars = ax.bar(
        x + width / 2, user_vals, width,
        color=COLOR_USER, alpha=0.9, label="Your Predicted Salary"
    )

    # Value labels
    for bar in market_bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1500,
                f"${h/1000:.0f}K",
                ha="center", va="bottom",
                fontsize=9, color="white", fontweight="bold"
            )

    for bar in user_bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1500,
                f"${h/1000:.0f}K",
                ha="center", va="bottom",
                fontsize=9, color=COLOR_USER, fontweight="bold"
            )

    # Styling
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Annual Salary (USD)", color="white", fontsize=11)
    ax.set_title(
        "Your Predicted Salary vs Market Average by Experience Level",
        color="white", fontsize=13, fontweight="bold", pad=15
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K")
    )
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.yaxis.label.set_color("white")
    ax.legend(facecolor="#1e2130", edgecolor="#444",
              labelcolor="white", fontsize=10)

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    logger.info("Chart built successfully.")
    return encoded