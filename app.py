"""
Medical Image Intelligence & Analytics
========================================
AI-powered analysis and insights from medical images using Groq Vision API.

Run:  streamlit run app.py
"""

import html
import io
import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from utils.image_processing import (
    validate_image,
    load_and_preprocess,
    compute_statistics,
    get_pixel_histogram,
    pil_to_bytes,
)
from utils.groq_client import get_groq_description
from utils.visualisations import (
    plot_histogram,
    plot_intensity_heatmap,
    plot_class_distribution,
    plot_sample_stats,
    plot_pixel_distribution_by_class,
)
from utils.dataset_analytics import generate_sample_dataset, compute_dataset_summary
from utils.reporting import (
    build_analysis_report,
    dataframe_to_csv_bytes,
    format_stats_for_display,
)

load_dotenv(override=True)

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Image Intelligence & Analytics",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Design system (CSS)
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');

    :root {
        --bg:        #0B1120;
        --bg-soft:   #0F172A;
        --surface:   #131C31;
        --surface-2: #1A253D;
        --border:    #243049;
        --border-hi: #34466B;
        --text:      #E6EBF5;
        --muted:     #8A99B8;
        --faint:     #5B6B8C;
        --primary:   #5B9DFF;
        --primary-2: #7C6CFF;
        --accent:    #2DD4BF;
        --warn:      #FBBF24;
        --danger:    #F87171;
        --radius:    16px;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Canvas ── */
    .stApp {
        background:
            radial-gradient(900px 500px at 85% -10%, rgba(124,108,255,0.14), transparent 60%),
            radial-gradient(700px 400px at -5% 0%, rgba(91,157,255,0.12), transparent 55%),
            var(--bg);
        color: var(--text);
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }
    header[data-testid="stHeader"] { background: transparent; }
    h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111A30 0%, #0B1120 100%);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] * { color: #C5D0E6; }

    .brand { display:flex; align-items:center; gap:12px; margin: 4px 0 18px; }
    .brand-logo {
        width: 42px; height: 42px; border-radius: 12px;
        display:flex; align-items:center; justify-content:center; font-size: 1.4rem;
        background: linear-gradient(135deg, var(--primary), var(--primary-2));
        box-shadow: 0 6px 20px rgba(91,157,255,0.35);
    }
    .brand-name { font-weight: 700; font-size: 0.98rem; color: #fff !important; line-height:1.15; }
    .brand-sub  { font-size: 0.72rem; color: var(--muted) !important; }

    .side-label {
        font-size: 0.68rem; letter-spacing: .12em; text-transform: uppercase;
        color: var(--faint) !important; font-weight: 600; margin: 18px 0 8px;
    }
    .status-row {
        display:flex; justify-content:space-between; align-items:center;
        background: var(--surface); border:1px solid var(--border);
        border-radius: 10px; padding: 9px 12px; margin-bottom: 8px; font-size: .8rem;
    }
    .pill {
        font-size:.7rem; font-weight:600; padding: 2px 9px; border-radius: 999px;
        border:1px solid transparent;
    }
    .pill-ok   { background: rgba(45,212,191,.12); color: var(--accent) !important; border-color: rgba(45,212,191,.4); }
    .pill-bad  { background: rgba(248,113,113,.12); color: var(--danger) !important; border-color: rgba(248,113,113,.4); }
    .mono { font-family: 'JetBrains Mono', monospace; font-size: .74rem; color: var(--muted) !important; }

    .how-step { display:flex; gap:10px; align-items:flex-start; margin-bottom: 10px; font-size:.8rem; }
    .how-num {
        flex: 0 0 22px; height:22px; border-radius:50%;
        background: var(--surface-2); border:1px solid var(--border-hi);
        display:flex; align-items:center; justify-content:center;
        font-size:.7rem; font-weight:700; color: var(--primary) !important;
    }

    /* ── Hero ── */
    .hero {
        position: relative; overflow: hidden;
        border: 1px solid var(--border); border-radius: 22px;
        padding: 30px 34px; margin-bottom: 22px;
        background: linear-gradient(135deg, rgba(91,157,255,0.12), rgba(124,108,255,0.10) 55%, rgba(45,212,191,0.07));
    }
    .hero::after {
        content:""; position:absolute; right:-60px; top:-60px; width:240px; height:240px;
        background: radial-gradient(circle, rgba(124,108,255,.35), transparent 70%);
        filter: blur(10px);
    }
    .hero h1 {
        margin: 0 0 6px; font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(90deg, #8FC0FF, #B8A8FF 60%, #7CF0DF);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { margin: 0; color: var(--muted); font-size: .98rem; max-width: 720px; }
    .badges { display:flex; gap:8px; margin-top: 16px; flex-wrap: wrap; position: relative; z-index: 1; }
    .badge {
        font-size:.72rem; font-weight:600; padding: 5px 12px; border-radius: 999px;
        background: rgba(255,255,255,.05); border:1px solid var(--border-hi); color: #C5D0E6;
    }

    /* ── Section headings ── */
    .sec { margin: 6px 0 14px; }
    .sec-title { font-size: 1.12rem; font-weight: 700; color: var(--text); display:flex; align-items:center; gap:8px; }
    .sec-sub   { font-size: .84rem; color: var(--muted); margin-top: 2px; }

    .mini-title {
        font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
        color: var(--muted); font-weight:600; margin-bottom: 10px;
    }

    /* ── Bordered containers become cards ── */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--surface);
        border-color: var(--border) !important;
        border-radius: var(--radius) !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px 18px;
        transition: border-color .2s, transform .2s;
    }
    [data-testid="stMetric"]:hover { border-color: var(--primary); transform: translateY(-1px); }
    [data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: .74rem !important; }
    [data-testid="stMetricLabel"] p { font-size: .74rem !important; letter-spacing: .04em; }
    [data-testid="stMetricValue"] { color: #fff !important; font-size: 1.45rem !important; font-weight: 700 !important; }

    /* ── Stat tiles ── */
    .stat-grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; }
    .stat-tile {
        background: var(--surface-2); border:1px solid var(--border);
        border-radius: 12px; padding: 12px 14px;
    }
    .stat-k { font-size:.72rem; color: var(--muted); margin-bottom: 4px; }
    .stat-v { font-size:1rem; font-weight:600; color: var(--primary); font-family:'JetBrains Mono', monospace; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface); border:1px solid var(--border);
        border-radius: 14px; padding: 5px; gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; color: var(--muted); font-weight: 600;
        padding: 8px 18px; height: auto;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--text); }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary), var(--primary-2)) !important;
        color: #fff !important;
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 18px; }

    /* ── Buttons ── */
    .stButton > button, [data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--primary), var(--primary-2));
        color: #fff; border: none; border-radius: 12px;
        font-weight: 600; padding: .6rem 1.4rem;
        box-shadow: 0 6px 18px rgba(91,157,255,.25);
        transition: transform .15s, box-shadow .15s, opacity .15s;
    }
    .stButton > button:hover {
        transform: translateY(-1px); box-shadow: 0 10px 24px rgba(91,157,255,.35);
        color:#fff; border:none;
    }
    .stButton > button:active { transform: translateY(0); }

    [data-testid="stDownloadButton"] button {
        background: transparent; color: var(--accent);
        border: 1px solid var(--accent); border-radius: 12px; font-weight: 600;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: rgba(45,212,191,.12); color: var(--accent); border-color: var(--accent);
    }

    /* ── Uploader ── */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-soft);
        border: 2px dashed var(--border-hi);
        border-radius: 14px;
        transition: border-color .2s, background .2s;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary); background: rgba(91,157,255,.05);
    }

    /* ── Inputs ── */
    [data-baseweb="select"] > div { background: var(--bg-soft); border-color: var(--border-hi); border-radius: 10px; }

    /* ── Expanders / alerts / images ── */
    [data-testid="stExpander"] { border:1px solid var(--border); border-radius: 12px; background: var(--surface); }
    .stAlert { border-radius: 12px; }
    img { border-radius: 12px; }
    hr { border-color: var(--border) !important; }

    /* ── Empty state ── */
    .empty {
        text-align:center; padding: 54px 20px; border-radius: 18px;
        background: linear-gradient(180deg, var(--surface), var(--bg-soft));
        border: 1px dashed var(--border-hi);
    }
    .empty .ico { font-size: 3.2rem; margin-bottom: 6px; }
    .empty h3 { margin: 6px 0 4px; font-size: 1.15rem; }
    .empty p  { color: var(--muted); font-size:.9rem; margin:0; }

    /* ── Disclaimer ── */
    .disclaimer {
        background: rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.45);
        border-radius: 12px; padding: 11px 14px; font-size:.8rem; color: #FDE68A;
    }
    [data-testid="stSidebar"] .disclaimer, [data-testid="stSidebar"] .disclaimer * { color:#FDE68A !important; }

    /* ── AI report ── */
    .ai-head {
        display:flex; align-items:center; gap:10px; margin-bottom: 4px;
        font-weight:700; font-size: 1rem;
    }
    .ai-dot { width:9px; height:9px; border-radius:50%; background: var(--accent); box-shadow: 0 0 10px var(--accent); }

    /* ── Footer ── */
    .footer { text-align:center; color: var(--faint); font-size:.78rem; padding: 14px 0 6px; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────
def section(title: str, subtitle: str = "", icon: str = ""):
    sub = f"<div class='sec-sub'>{html.escape(subtitle)}</div>" if subtitle else ""
    st.markdown(
        f"<div class='sec'><div class='sec-title'>{icon} {html.escape(title)}</div>{sub}</div>",
        unsafe_allow_html=True,
    )


def mini_title(text: str):
    st.markdown(f"<div class='mini-title'>{html.escape(text)}</div>", unsafe_allow_html=True)


def stat_tiles(df: pd.DataFrame):
    tiles = "".join(
        f"<div class='stat-tile'><div class='stat-k'>{html.escape(str(r['Metric']))}</div>"
        f"<div class='stat-v'>{html.escape(str(r['Value']))}</div></div>"
        for _, r in df.iterrows()
    )
    st.markdown(f"<div class='stat-grid'>{tiles}</div>", unsafe_allow_html=True)


CHART_CFG = {"displayModeBar": False}


# ──────────────────────────────────────────────
# Configuration from environment
# ──────────────────────────────────────────────
api_key = os.getenv("GROQ_API_KEY", "")
raw_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
INVALID_MODELS = {
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "llama-3.2-11b-vision-preview",
    "llama-3.2-90b-vision-preview",
}
selected_model = "qwen/qwen3.8-27b" if (not raw_model or raw_model in INVALID_MODELS) else raw_model
api_ready = bool(api_key) and not api_key.startswith("your_groq")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div class='brand'>
            <div class='brand-logo'>🩺</div>
            <div>
                <div class='brand-name'>MedImage Intelligence</div>
                <div class='brand-sub'>Analytics workspace</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


    st.markdown("<div class='side-label'>Sample dataset</div>", unsafe_allow_html=True)
    n_samples = st.slider(
        "Synthetic samples", min_value=50, max_value=500, value=100, step=50,
        help="Controls the size of the synthetic dataset on the Analytics tab.",
    )

    st.markdown("<div class='side-label'>How it works</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='how-step'><div class='how-num'>1</div><div>Upload a chest X-ray or pick a built-in sample.</div></div>
        <div class='how-step'><div class='how-num'>2</div><div>Review image statistics and pixel distributions.</div></div>
        <div class='how-step'><div class='how-num'>3</div><div>Run the AI description and export the report.</div></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='disclaimer'>⚠️ <b>Educational prototype.</b><br>Not for clinical use.</div>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Hero
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class='hero'>
        <h1>Medical Image Intelligence &amp; Analytics</h1>
        <p>Inspect chest X-rays, explore pixel-level statistics, and generate AI-assisted
        descriptions powered by the Groq Vision API.</p>
        <div class='badges'>
            <span class='badge'>⚡ Groq Vision</span>
            <span class='badge'>📈 Interactive analytics</span>
            <span class='badge'>📥 CSV export</span>
            <span class='badge'>🎓 Educational use</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────
# Main Tabs
# ──────────────────────────────────────────────
tab_analysis, tab_dataset = st.tabs(["🔬  Image Analysis", "📊  Dataset Analytics"])


# ╔══════════════════════════════════════════════╗
# ║            TAB 1 — IMAGE ANALYSIS           ║
# ╚══════════════════════════════════════════════╝
with tab_analysis:

    section("Choose an image", "Upload your own chest X-ray or try a curated sample.", "📤")

    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    sample_files = []
    if os.path.exists(sample_dir):
        sample_files = sorted(
            f for f in os.listdir(sample_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

    with st.container(border=True):
        col_upload, col_sample = st.columns([3, 2], gap="large")
        with col_upload:
            uploaded_file = st.file_uploader(
                "Drag and drop a file here, or click to browse",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=False,
                help="JPG or PNG, up to 10 MB.",
            )

        selected_sample = None
        with col_sample:
            if sample_files:
                sample_options = ["None (Upload custom)"] + sample_files
                choice = st.selectbox(
                    "🧪 Or pick a built-in sample",
                    options=sample_options,
                    index=0,
                    help="Test the dashboard instantly using curated sample chest X-rays.",
                )
                if choice != "None (Upload custom)":
                    selected_sample = choice
                st.caption("A custom upload always takes priority over a sample.")
            else:
                st.caption("No built-in samples found in `sample_images/`.")

    # Determine active file (custom upload takes priority)
    active_file = uploaded_file
    if active_file is None and selected_sample:
        sample_path = os.path.join(sample_dir, selected_sample)
        with open(sample_path, "rb") as sf:
            content = sf.read()
        buf = io.BytesIO(content)
        buf.name = selected_sample
        buf.size = len(content)
        active_file = buf

    if active_file is None:
        st.markdown(
            """
            <div class='empty'>
                <div class='ico'>🫁</div>
                <h3>No image selected yet</h3>
                <p>Upload a chest X-ray (JPG / PNG, max 10 MB) or choose a built-in sample to begin.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    uploaded_file = active_file

    # ── Validation ───────────────────────────────
    is_valid, msg = validate_image(uploaded_file)
    if not is_valid:
        st.error(f"❌ {msg}")
        st.stop()

    st.success(f"✅ {msg}  ·  **{uploaded_file.name}**  ({uploaded_file.size / 1024:.1f} KB)")

    # ── Pre-process ──────────────────────────────
    with st.spinner("Preprocessing image…"):
        img_data = load_and_preprocess(uploaded_file)
        stats = compute_statistics(img_data, uploaded_file.size)
        hist_counts, hist_bins = get_pixel_histogram(img_data)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Key metrics ──────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Width", f"{stats['Width (px)']} px")
    k2.metric("Height", f"{stats['Height (px)']} px")
    k3.metric("Mean Intensity", stats["Mean Intensity"])
    k4.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Workspace: image viewer + insights ───────
    col_view, col_insight = st.columns([5, 7], gap="large")

    with col_view:
        with st.container(border=True):
            mini_title("Image viewer")
            v_orig, v_gray, v_eq = st.tabs(["Original", "Grayscale", "Equalised"])
            with v_orig:
                st.image(img_data["original"], use_container_width=True, caption=uploaded_file.name)
            with v_gray:
                st.image(img_data["grayscale"], use_container_width=True, caption="Grayscale")
            with v_eq:
                st.image(
                    img_data["enhanced"],
                    use_container_width=True,
                    caption="Histogram equalised — improves contrast of lung fields",
                )

    with col_insight:
        with st.container(border=True):
            i_hist, i_heat, i_stats = st.tabs(["📊 Histogram", "🌡️ Intensity heatmap", "📐 Statistics"])
            with i_hist:
                st.plotly_chart(
                    plot_histogram(hist_counts, hist_bins),
                    use_container_width=True,
                    config=CHART_CFG,
                )
            with i_heat:
                st.plotly_chart(
                    plot_intensity_heatmap(img_data["array"]),
                    use_container_width=True,
                    config=CHART_CFG,
                )
            with i_stats:
                stat_tiles(format_stats_for_display(stats))

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ── AI Analysis ──────────────────────────────
    section(
        "AI-powered description",
        "Generate a structured, non-diagnostic description of the image with Groq Vision.",
        "🤖",
    )

    if "ai_result" not in st.session_state:
        st.session_state.ai_result = None
    if "last_filename" not in st.session_state:
        st.session_state.last_filename = None

    # Reset result when a new file is chosen
    if st.session_state.last_filename != uploaded_file.name:
        st.session_state.ai_result = None
        st.session_state.last_filename = uploaded_file.name

    with st.container(border=True):
        a_text, a_btn = st.columns([5, 2], gap="large", vertical_alignment="center")
        with a_text:
            st.markdown(
                f"<div class='ai-head'><span class='ai-dot'></span>Ready to analyse "
                f"<span class='mono'>{html.escape(uploaded_file.name)}</span></div>"
                "<div class='sec-sub'>The image is sent to the Groq Vision model for a descriptive summary. "
                "Results are not a clinical diagnosis.</div>",
                unsafe_allow_html=True,
            )
        with a_btn:
            analyse_btn = st.button("🔍 Analyse Image", use_container_width=True)

    if analyse_btn:
        if not api_ready:
            st.warning("⚠️ No valid Groq API key found. Please set `GROQ_API_KEY` in your `.env` file.")
        else:
            with st.spinner("Groq Vision is analysing the image…"):
                result = get_groq_description(
                    img=img_data["original"],
                    api_key=api_key,
                    model=selected_model,
                )
                st.session_state.ai_result = result

    if st.session_state.ai_result:
        result = st.session_state.ai_result
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        if result["success"]:
            i1, i2, i3 = st.columns(3)
            i1.metric("🤖 Model", result["model"].split("/")[-1][:25])
            i2.metric("🪙 Tokens Used", result["tokens_used"])
            i3.metric("⚡ Latency", f"{result['latency_ms']} ms")

            with st.container(border=True):
                mini_title("Generated description")
                st.markdown(result["description"])

            st.markdown(
                "<div class='disclaimer' style='margin-top:12px;'>"
                "⚠️ This description is AI-generated and is <b>not</b> a clinical diagnosis. "
                "A qualified healthcare professional must interpret all medical images."
                "</div>",
                unsafe_allow_html=True,
            )

            report_df = build_analysis_report(
                filename=uploaded_file.name,
                stats=stats,
                description=result["description"],
                model=result["model"],
                tokens=result["tokens_used"],
                latency_ms=result["latency_ms"],
            )
            csv_bytes = dataframe_to_csv_bytes(report_df)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.download_button(
                label="⬇️ Download Analysis Report (CSV)",
                data=csv_bytes,
                file_name=f"analysis_{uploaded_file.name}.csv",
                mime="text/csv",
            )
        else:
            st.error(result["description"])


# ╔══════════════════════════════════════════════╗
# ║          TAB 2 — DATASET ANALYTICS          ║
# ╚══════════════════════════════════════════════╝
with tab_dataset:
    section(
        "Dataset insights",
        "Explore class balance, image properties, and intensity patterns.",
        "📊",
    )
    st.info(
        "These charts use a **synthetic sample dataset** that mirrors the Kaggle "
        "Chest X-Ray Images (Pneumonia) distribution. Get the real data from "
        "[Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).",
        icon="ℹ️",
    )

    with st.spinner("Generating sample dataset…"):
        sample_df = generate_sample_dataset(n=n_samples)
        summary_df = compute_dataset_summary(sample_df)

    # ── Overview metrics ──────────────────────────
    n_total = len(sample_df)
    n_normal = int((sample_df["label"] == "NORMAL").sum())
    n_pneu = int((sample_df["label"] == "PNEUMONIA").sum())

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Total Images", n_total)
    d2.metric("NORMAL", n_normal, f"{n_normal / max(n_total, 1):.0%} of set", delta_color="off")
    d3.metric("PNEUMONIA", n_pneu, f"{n_pneu / max(n_total, 1):.0%} of set", delta_color="off")
    d4.metric("Avg File Size", f"{sample_df['file_size_kb'].mean():.0f} KB")

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Charts ────────────────────────────────────
    r1c1, r1c2 = st.columns(2, gap="medium")
    with r1c1:
        with st.container(border=True):
            mini_title("Class distribution")
            st.plotly_chart(
                plot_class_distribution(sample_df),
                use_container_width=True,
                config=CHART_CFG,
            )
    with r1c2:
        with st.container(border=True):
            mini_title("Image dimensions & size")
            st.plotly_chart(
                plot_sample_stats(sample_df),
                use_container_width=True,
                config=CHART_CFG,
            )

    with st.container(border=True):
        mini_title("Mean pixel intensity by class")
        st.plotly_chart(
            plot_pixel_distribution_by_class(sample_df),
            use_container_width=True,
            config=CHART_CFG,
        )

    # ── Data table & download ─────────────────────
    with st.expander("🗃️ Sample image stats table (first 20 rows)"):
        st.dataframe(sample_df.head(20), use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download Full Sample Dataset (CSV)",
        data=dataframe_to_csv_bytes(sample_df),
        file_name="sample_chest_xray_dataset.csv",
        mime="text/csv",
    )


# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div class='footer'>
        Medical Image Intelligence &amp; Analytics · Built with
        <span style='color:#5B9DFF;'>Streamlit</span> +
        <span style='color:#9C8CFF;'>Groq Vision API</span> ·
        <span style='color:#F472B6;'>Educational use only — not for clinical diagnosis</span>
    </div>
    """,
    unsafe_allow_html=True,
)