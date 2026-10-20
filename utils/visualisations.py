"""
Visualisation helpers – Plotly charts for the Streamlit dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Colour palette (consistent across all charts)
# ---------------------------------------------------------------------------
PRIMARY   = "#4F8EF7"
SECONDARY = "#34D399"
ACCENT    = "#F472B6"
BG_DARK   = "#0F172A"
BG_CARD   = "#1E293B"
TEXT      = "#E2E8F0"
GRID      = "#334155"


def plot_histogram(counts: np.ndarray, bins: np.ndarray) -> go.Figure:
    """Grayscale pixel-intensity histogram."""
    bin_centres = (bins[:-1] + bins[1:]) / 2

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=bin_centres,
            y=counts,
            marker=dict(
                color=bin_centres,
                colorscale="Blues",
                showscale=False,
            ),
            name="Pixel Count",
        )
    )
    fig.update_layout(
        title=dict(text="Grayscale Histogram", font=dict(color=TEXT, size=14)),
        xaxis=dict(title="Pixel Intensity (0-255)", color=TEXT, gridcolor=GRID),
        yaxis=dict(title="Count", color=TEXT, gridcolor=GRID),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT),
        margin=dict(l=40, r=20, t=50, b=40),
        height=280,
    )
    return fig


def plot_intensity_heatmap(arr: np.ndarray) -> go.Figure:
    """2-D heatmap of pixel intensities (downsampled for speed)."""
    # Downsample to 128×128 for performance
    from PIL import Image as PILImage
    import io

    pil = PILImage.fromarray(arr.astype(np.uint8))
    pil = pil.resize((128, 128))
    arr_small = np.array(pil)

    fig = go.Figure(
        go.Heatmap(z=arr_small, colorscale="Plasma", showscale=True)
    )
    fig.update_layout(
        title=dict(text="Intensity Heatmap", font=dict(color=TEXT, size=14)),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT),
        margin=dict(l=40, r=20, t=50, b=40),
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def plot_class_distribution(df: pd.DataFrame, label_col: str = "label") -> go.Figure:
    """Pie chart for dataset class distribution."""
    counts = df[label_col].value_counts().reset_index()
    counts.columns = ["Class", "Count"]

    colours = [PRIMARY, SECONDARY, ACCENT, "#FBBF24", "#A78BFA"]

    fig = go.Figure(
        go.Pie(
            labels=counts["Class"],
            values=counts["Count"],
            hole=0.5,
            marker=dict(colors=colours[: len(counts)], line=dict(color=BG_DARK, width=2)),
            textinfo="label+percent",
            textfont=dict(color=TEXT, size=12),
        )
    )
    fig.update_layout(
        title=dict(text="Class Distribution", font=dict(color=TEXT, size=14)),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT),
        legend=dict(font=dict(color=TEXT)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=300,
    )
    return fig


def plot_sample_stats(df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing avg width, height, file size across classes."""
    agg = (
        df.groupby("label")[["width", "height", "file_size_kb"]]
        .mean()
        .reset_index()
    )

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Avg Width (px)", "Avg Height (px)", "Avg Size (KB)"],
    )

    cols_map = {1: "width", 2: "height", 3: "file_size_kb"}
    clrs = [PRIMARY, SECONDARY, ACCENT]

    for col_idx, metric in cols_map.items():
        fig.add_trace(
            go.Bar(
                x=agg["label"],
                y=agg[metric],
                marker_color=clrs[col_idx - 1],
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT),
        margin=dict(l=20, r=20, t=60, b=40),
        height=320,
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=GRID, color=TEXT)
    return fig


def plot_pixel_distribution_by_class(df: pd.DataFrame) -> go.Figure:
    """Box plot of mean pixel intensity grouped by class label."""
    if "mean_intensity" not in df.columns:
        return go.Figure()

    fig = px.box(
        df,
        x="label",
        y="mean_intensity",
        color="label",
        color_discrete_sequence=[PRIMARY, SECONDARY, ACCENT],
        points="all",
    )
    fig.update_layout(
        title=dict(text="Mean Pixel Intensity by Class", font=dict(color=TEXT, size=14)),
        paper_bgcolor=BG_CARD,
        plot_bgcolor=BG_CARD,
        font=dict(color=TEXT),
        xaxis=dict(title="Class", color=TEXT, gridcolor=GRID),
        yaxis=dict(title="Mean Intensity", color=TEXT, gridcolor=GRID),
        legend=dict(font=dict(color=TEXT)),
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40),
        height=320,
    )
    return fig
