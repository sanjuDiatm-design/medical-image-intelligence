"""
Groq Vision API Client
Sends medical images to Groq's vision LLM and returns structured descriptions.
"""

import os
import time
import base64
import io
from typing import Optional

from dotenv import load_dotenv
from groq import Groq
from PIL import Image

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INVALID_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
}

_env_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
DEFAULT_MODEL = "qwen/qwen3.8-27b" if (not _env_model or _env_model in INVALID_MODELS) else _env_model

SYSTEM_PROMPT = """You are an expert radiologist assistant. Analyse the provided medical image and
return a structured report with the following numbered sections:

1. Image Overview – modality, body region, projection
2. Visible Visual Characteristics – density, contrast, notable structures
3. Image Quality or Limitations – exposure, artefacts, noise
4. AI-Generated Observations – key findings (if any)

⚠️ DISCLAIMER: This description is AI-generated and NOT a clinical diagnosis.
A qualified healthcare professional must interpret all medical images."""

USER_PROMPT = """Please analyse this chest X-ray image and provide a detailed,
structured description following the format in your instructions."""


def _encode_pil_image(img: Image.Image, max_size: int = 1024) -> str:
    """
    Resize image to fit within max_size × max_size, then base64-encode as JPEG.
    Keeps payload small to stay within Groq's token limits.
    """
    img = img.convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def get_groq_description(
    img: Image.Image,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_retries: int = 2,
) -> dict:
    """
    Send a PIL image to Groq Vision and return the structured description.

    Returns
    -------
    dict with keys:
        "success"     : bool
        "description" : str  (AI text or error message)
        "model"       : str
        "tokens_used" : int
        "latency_ms"  : float
    """
    key = api_key or os.getenv("GROQ_API_KEY", "")
    if not key or key.startswith("your_groq"):
        return {
            "success": False,
            "description": "❌ No valid Groq API key found. Please add your key in the sidebar.",
            "model": model,
            "tokens_used": 0,
            "latency_ms": 0,
        }

    active_model = model or DEFAULT_MODEL
    if active_model in INVALID_MODELS:
        active_model = DEFAULT_MODEL

    client = Groq(api_key=key)
    b64_image = _encode_pil_image(img)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": USER_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"},
                },
            ],
        },
    ]

    for attempt in range(max_retries + 1):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=active_model,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            latency_ms = (time.time() - start) * 1000

            return {
                "success": True,
                "description": response.choices[0].message.content,
                "model": response.model,
                "tokens_used": response.usage.total_tokens,
                "latency_ms": round(latency_ms, 1),
            }

        except Exception as e:
            if attempt < max_retries:
                time.sleep(2 ** attempt)          # exponential back-off
            else:
                return {
                    "success": False,
                    "description": f"❌ Groq API error after {max_retries + 1} attempts: {e}",
                    "model": model,
                    "tokens_used": 0,
                    "latency_ms": 0,
                }
