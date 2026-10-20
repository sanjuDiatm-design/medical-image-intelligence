"""
Dataset Analytics helpers
Generates a synthetic-but-realistic sample dataset so the 'Dataset Analytics'
tab works even before the user downloads any Kaggle data.
"""

import random
import numpy as np
import pandas as pd


LABELS = ["NORMAL", "PNEUMONIA"]
LABEL_WEIGHTS = [0.70, 0.30]    # matches the Kaggle chest-X-ray distribution


def generate_sample_dataset(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Build a synthetic dataframe that mimics the Kaggle chest X-ray dataset
    metadata (image dimensions, intensity stats, file sizes).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    labels = rng.choice(LABELS, size=n, p=LABEL_WEIGHTS)

    widths  = rng.integers(800, 2000, size=n)
    heights = rng.integers(800, 2000, size=n)

    # Pneumonia images tend to have lower mean intensity (more opaque)
    mean_intensities = np.where(
        labels == "PNEUMONIA",
        rng.normal(110, 20, n),
        rng.normal(145, 15, n),
    ).clip(0, 255)

    std_devs = rng.uniform(40, 80, n)
    file_sizes_kb = (widths * heights * rng.uniform(0.3, 0.6, n) / 1024).astype(int)

    df = pd.DataFrame(
        {
            "filename": [f"img_{i:04d}.jpg" for i in range(n)],
            "label": labels,
            "width": widths,
            "height": heights,
            "aspect_ratio": (widths / heights).round(2),
            "mean_intensity": mean_intensities.round(2),
            "std_dev": std_devs.round(2),
            "file_size_kb": file_sizes_kb,
        }
    )
    return df


def compute_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-class summary statistics table."""
    numeric_cols = ["width", "height", "mean_intensity", "std_dev", "file_size_kb"]
    summary = (
        df.groupby("label")[numeric_cols]
        .agg(["mean", "std", "min", "max"])
        .round(1)
    )
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
    return summary.reset_index()
