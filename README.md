
# IT4868E - Topic: News Recommendation System

This repository implements a News Recommendation System using PyTorch and PyTorch Lightning. It features a high-performance data pipeline for the Ebnerd dataset and explores three distinct modeling approaches, ranging from a baseline NAML variant to complex transformer-based ranking architectures.

## 🚀 Key Features

* **High Performance**: Utilizes **Polars** for efficient preprocessing of large-scale Parquet datasets.
* **Baseline (NAML Variant)**: Enhanced news encoder using Multi-Head Self-Attention.
* **Method 1 (Quality-Aware)**: Disentangles user history into "High Quality" and "Low Quality" views based on dwell time and scroll depth to reduce noise.
* **Method 2 (Time-Feature & RankFormer)**: A sophisticated architecture employing **Sinusoidal Embeddings** for numerical features, **Multi-Interest Learning**, and a **Transformer Ranker** to model complex interactions.

## 📂 Project Structure

```text
.
├── pyproject.toml          # Project configuration and dependencies (uv managed)
├── run_baseline.sh         # Training script for Baseline
├── run_method1.sh          # Training script for Method 1 (Quality-Aware)
├── run_method2.sh          # Training script for Method 2 (RankFormer)
├── src
│   ├── baseline            # Baseline Model Implementation
│   ├── method1             # Quality-Aware Model Implementation
│   ├── method2             # Time-Feature & RankFormer Implementation
│   │   ├── model.py        # Contains SinusoidalEmbedding, MultiInterestUserEncoder, RankFormer
│   │   └── ...
│   └── preprocess
│       └── preprocess.py   # Polars-based data pipeline
└── ...

```

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast package management.

### 1. Clone the repository

```bash
git clone https://github.com/romkhonglun/wm20251
cd wm20251

```

### 2. Install Dependencies

```bash
uv sync

```

*Note: Dependencies are managed via `pyproject.toml`.*

### ⚠️ Important: PyTorch & CUDA Version

By default, the environment might install a CPU-only version of PyTorch or a CUDA version incompatible with your hardware. **To ensure GPU acceleration**, please verify your CUDA version (`nvidia-smi`) and install the appropriate PyTorch version manually:

```bash
# Example for CUDA 11.8
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Example for CUDA 12.1
uv pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```

## 📊 Data Preparation (Ebnerd Dataset)

1. **Place Raw Data**: Ensure your raw parquet files are located in the input directory (default: `input/ebnerd_testset` inside the configured root).
2. **Run Preprocessing**:
This script handles ID mapping, log-normalization of numerical features, and train/test splitting.
```bash
uv run src/preprocess/preprocess.py

```




## 🏃 Training Instructions

*Please update the paths (e.g., `/home2/congnh/...`) in the `.sh` files to match your local environment before running.*

### Train Baseline

```bash
bash run_baseline.sh

```

### Train Method 1 (Quality-Aware)

```bash
bash run_method1.sh

```

### Train Method 2 (RankFormer)

Requires pre-trained body embeddings (stored as `.npy`).

```bash
bash run_method2.sh

```

*Arguments used: `--embedding-path` pointing to the `.npy` file.*
