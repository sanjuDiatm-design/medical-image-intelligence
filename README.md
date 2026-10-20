# Medical Image Intelligence & Analytics

An end-to-end data science project combining image preprocessing, exploratory data analysis,
and vision-language model integration to analyze chest X-ray images and generate AI-assisted
descriptions and useful analytics.

---

## 🚀 Quick Start

### 1. Clone / download the project

```bash
git clone https://github.com/your-username/medical-image-intelligence.git
cd medical-image-intelligence
```

### 2. Create & activate virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `.env` file (copy from `.env.example`):

```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

Get a free API key at [console.groq.com](https://console.groq.com).

### 5. Run the app

```bash
streamlit run app.py
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 📤 Upload & Validate | Support JPG / PNG up to 10 MB with format & integrity checks |
| 🔧 Preprocessing | Grayscale conversion, histogram equalisation, normalisation |
| 📐 Statistics | Width, height, aspect ratio, mean intensity, std dev, file size |
| 📊 Visualisations | Pixel histogram, intensity heatmap, interactive charts |
| 🤖 AI Description | Groq Vision LLM generates structured radiology-style reports |
| 📊 Dataset Analytics | Synthetic dataset EDA (class distribution, intensity, dimensions) |
| 📥 Export | Download analysis reports as CSV |

---

## 🏗️ Architecture

```
Chest X-ray (JPG/PNG)
        │
        ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Image Upload   │────▶│  Preprocessing   │────▶│   Groq Vision API   │
│  & Validation   │     │  & EDA (Pillow,  │     │  (Qwen 3.8 27B /    │
│                 │     │  NumPy, Plotly)  │     │   Vision multimodal)│
└─────────────────┘     └──────────────────┘     └─────────────────────┘
                                                           │
                                                           ▼
                                              ┌────────────────────────┐
                                              │  Analytics & Reporting │
                                              │  (Streamlit dashboard, │
                                              │  CSV export)           │
                                              └────────────────────────┘
```

---

## 📂 Project Structure

```
ImgProject/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .streamlit/
│   └── config.toml                 # Streamlit dark-theme config
└── utils/
    ├── __init__.py
    ├── image_processing.py         # Preprocessing & statistics
    ├── groq_client.py              # Groq Vision API client
    ├── visualisations.py           # Plotly chart helpers
    ├── dataset_analytics.py        # Synthetic EDA dataset
    └── reporting.py                # CSV export utilities
```

---

## 📊 Dataset

**Chest X-Ray Images (Pneumonia)** — Kaggle  
~5,863 images labelled NORMAL & PNEUMONIA  
[Download here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

> For initial development, use a small subset (50–100 images).
> Do NOT upload the full dataset to GitHub (respect Kaggle terms).

---

## ⚠️ Important Notice

This is an **educational prototype**, not a medical device.  
AI-generated descriptions may be inaccurate or incomplete.  
Pixel statistics are not clinical measurements.  
**Do not use this tool for diagnosis or treatment decisions.**

---

## 📈 Evaluation Metrics

| Metric | What to Measure |
|---|---|
| API Success Rate | Successful responses ÷ total requests |
| Avg. Latency | Mean time per image analysis |
| Image Validation Rate | Valid uploads ÷ total uploads |
| Description Completeness | % outputs with all required sections |
| Human-rated Usefulness | Score from manual review (0–2) |
| Reproducibility | Compare outputs across repeated runs |
| Cost Efficiency | Track token usage and API costs |
