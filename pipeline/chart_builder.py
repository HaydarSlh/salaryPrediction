"""
chart_builder.py
----------------
Takes chart_data returned by Gemini and builds a Matplotlib figure.
Returns the chart as a base64-encoded PNG string for storage in Supabase.
"""

import io
import base64
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)

# ── Colors ────────────────────────────────────────────────────────────────────
COLOR_MARKET = "#4C72B0"   # blue  — market average bars
COLOR_USER   = "#DD8452"   # coral — user's bar


def build_salary_chart(chart_data: dict) -> str:
    """
    Builds a grouped bar chart: market average vs user salary by experience level.

    Parameters
    ----------
    chart_data : dict
        As returned by Gemini — keys: title, labels, market_values,
        user_experience, user_salary

    Returns
    -------
    str
        Base64-encoded PNG image string.
    """
    labels        = chart_data["labels"]
    market_values = chart_data["market_values"]
    user_exp      = chart_data["user_experience"]
    user_salary   = chart_data["user_salary"]
    title         = chart_data.get("title", "Salary vs Market Average")

    x      = np.arange(len(labels))
    width  = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # ── Market average bars ───────────────────────────────────────────────
    market_bars = ax.bar(
        x - width / 2, market_values, width,
        color=COLOR_MARKET, alpha=0.85, label="Market Average"
    )

    # ── User salary bar — only on their experience level ──────────────────
    user_values = [0] * len(labels)
    if user_exp in labels:
        user_idx = labels.index(user_exp)
        user_values[user_idx] = user_salary

    user_bars = ax.bar(
        x + width / 2, user_values, width,
        color=COLOR_USER, alpha=0.9, label="Your Predicted Salary"
    )

    # ── Value labels on bars ──────────────────────────────────────────────
    for bar in market_bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1500,
                f"${h/1000:.0f}K",
                ha="center", va="bottom", fontsize=9,
                color="white", fontweight="bold"
            )

    for bar in user_bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 1500,
                f"${h/1000:.0f}K",
                ha="center", va="bottom", fontsize=9,
                color=COLOR_USER, fontweight="bold"
            )

    # ── Styling ───────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylabel("Annual Salary (USD)", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda val, _: f"${val/1000:.0f}K")
    )
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#444")
    ax.yaxis.label.set_color("white")

    # ── Legend ────────────────────────────────────────────────────────────
    legend = ax.legend(
        facecolor="#1e2130", edgecolor="#444",
        labelcolor="white", fontsize=10
    )

    plt.tight_layout()

    # ── Encode to base64 ──────────────────────────────────────────────────
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=150,
                bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)

    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    logger.info("Chart built and encoded successfully.")
    return encoded
