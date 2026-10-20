"""
Export & Reporting helpers
Produces CSV and text reports from analysis results.
"""

import io
import csv
import datetime
import pandas as pd


def build_analysis_report(
    filename: str,
    stats: dict,
    description: str,
    model: str,
    tokens: int,
    latency_ms: float,
) -> pd.DataFrame:
    """Combine image stats and AI description into a single-row DataFrame."""
    row = {
        "filename": filename,
        "analysis_timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "groq_model": model,
        "tokens_used": tokens,
        "latency_ms": latency_ms,
        **stats,
        "ai_description": description.replace("\n", " "),
    }
    return pd.DataFrame([row])


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialise a DataFrame to UTF-8 CSV bytes for st.download_button."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")


def format_stats_for_display(stats: dict) -> pd.DataFrame:
    """Convert the stats dict to a two-column 'Metric / Value' DataFrame."""
    return pd.DataFrame(
        [{"Metric": k, "Value": v} for k, v in stats.items()]
    )
