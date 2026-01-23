# ML Rust Service — High-Performance ML Inference API

![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)
![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.16%2B-green)
![Axum](https://img.shields.io/badge/Axum-0.7-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready, high-performance **Machine Learning inference service** built with **Rust** and **Axum**, powered by **ONNX Runtime**.  
This service provides a REST API for **sentiment analysis** trained on the Kaggle IMDB dataset, complemented by a **Streamlit web interface**.

---

# Kaggel
![IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data)

## About Dataset

  IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
  This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.
  For more dataset information, please go through the following link,
  http://ai.stanford.edu/~amaas/data/sentiment/

## Features

- **Blazing Fast Inference** — Rust performance with ONNX Runtime optimizations  
- **Modern Async API** — High‑concurrency handling with Axum  
- **Easy Deployment** — Batch scripts for Windows development & startup  
- **Full ML Pipeline** — From training to serving  
- **Interactive UI** — Streamlit frontend for testing predictions

---

## Prerequisites

- **Rust 1.70+** — https://www.rust-lang.org/tools/install  
- **Python 3.8+** — required for training & Streamlit UI  
- **ONNX Runtime** — installed automatically via Python dependencies  
- **Kaggle Account** (optional) — for downloading the IMDB dataset

---

## Project Structure

```
ml-rust-service/
├── python_ml/          # Python model training scripts
├── rust_api/           # Rust Axum API server
├── streamlit_ui/       # Streamlit web interface
├── *.bat               # Windows automation scripts
└── .gitignore
```

---

## Quick Start (Windows)

Run the full pipeline with minimal setup.

### 1. Train the model (Python)

```bat
train_model.bat
```

### 2. Start the API server (Rust)

```bat
run_api.bat
```

### 3. Launch the web interface

```bat
run_frontend.bat
```

### One‑command startup (all services)

```bat
start_all.bat
```

---

## Model Training

The sentiment model is trained on the **Kaggle IMDB Dataset**.

```bash
cd python_ml
pip install -r requirements.txt
python train_model.py
```

Training outputs an optimized **ONNX model** used by the Rust inference service.

---

## 🔌 API Usage

Base URL:

```
http://localhost:3000
```

### ➤ Analyze Sentiment

**Request**

```http
POST /analyze
Content-Type: application/json
```

```json
{
  "text": "This movie was absolutely fantastic!"
}
```

**Response**

```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "text": "This movie was absolutely fantastic!"
}
```

---

### ➤ Health Check

**Request**

```http
GET /health
```

**Response**

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## 🌐 Web Interface

Streamlit UI available at:

```
http://localhost:8501
```

Features:

- Interactive text input  
- Real‑time predictions  
- Confidence scores  
- Positive / negative visualization  
- Batch text processing

---

## Development

### Rust API

```bash
cd rust_api
cargo build
cargo run
```

### Streamlit UI

```bash
cd streamlit_ui
streamlit run app.py
```

---

## Performance

- **Inference latency:** `< 10 ms` per request (typical hardware)  
- **Concurrency:** Hundreds of parallel connections supported  
- **Memory usage:** Efficient tensor handling via ONNX Runtime

---

## Configuration

Optional environment variables:

| Variable     | Description             | Default     |
|--------------|-------------------------|-------------|
| `API_PORT`   | Rust API port           | `3000`      |
| `MODEL_PATH` | Path to ONNX model      | `model.onnx`|
| `LOG_LEVEL`  | Logging verbosity       | `info`      |

