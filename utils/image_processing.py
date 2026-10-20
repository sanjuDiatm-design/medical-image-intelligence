"""
Image Processing & EDA Utilities
Handles image loading, preprocessing, validation, and statistical analysis.
"""

import io
import base64
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import pandas as pd


ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE_MB = 10


def validate_image(uploaded_file) -> tuple[bool, str]:
    """Validate uploaded image for format and file size."""
    ext = uploaded_file.name.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported format '{ext}'. Please upload JPG or PNG."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB."

    try:
        img = Image.open(uploaded_file)
        img.verify()
        uploaded_file.seek(0)
    except Exception as e:
        return False, f"Invalid or corrupted image file: {e}"

    uploaded_file.seek(0)
    return True, "Image is valid."


def load_and_preprocess(uploaded_file) -> dict:
    """
    Load an image and return both the PIL image and a preprocessed
    (resized + normalized) version for downstream analysis.
    """
    uploaded_file.seek(0)
    img_original = Image.open(uploaded_file).convert("RGB")

    # Preprocessing pipeline
    img_gray = img_original.convert("L")                        # Grayscale
    img_resized = img_gray.resize((512, 512), Image.LANCZOS)    # Standardise size
    img_enhanced = ImageOps.equalize(img_resized)               # Histogram equalisation

    arr = np.array(img_resized, dtype=np.float32)
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # [0,1] normalise

    return {
        "original": img_original,
        "grayscale": img_gray,
        "preprocessed": img_resized,
        "enhanced": img_enhanced,
        "array": arr,
        "array_norm": arr_norm,
    }


def compute_statistics(img_data: dict, file_size_bytes: int) -> dict:
    """Compute pixel-level statistics from the processed image."""
    arr = img_data["array"]
    original = img_data["original"]
    width, height = original.size
    aspect = round(width / height, 2)

    return {
        "Width (px)": width,
        "Height (px)": height,
        "Aspect Ratio": aspect,
        "Mean Intensity": round(float(arr.mean()), 2),
        "Std Dev": round(float(arr.std()), 2),
        "Min Pixel": int(arr.min()),
        "Max Pixel": int(arr.max()),
        "File Size (KB)": round(file_size_bytes / 1024, 1),
    }


def get_pixel_histogram(img_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return histogram values and bin edges for the grayscale image."""
    arr = img_data["array"].flatten().astype(np.uint8)
    counts, bins = np.histogram(arr, bins=64, range=(0, 256))
    return counts, bins


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string for API submission."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Convert a PIL image to raw bytes."""
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return buffer.getvalue()
