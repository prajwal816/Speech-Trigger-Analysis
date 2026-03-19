# Machine Learning & NLP Trigger Analysis System

A production-grade system for analysing speech/audio features and identifying trigger patterns — with full model explainability.

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Feature Engineering Strategy](#feature-engineering-strategy)
5. [Model Architecture](#model-architecture)
6. [Explainability Insights](#explainability-insights)
7. [Performance Results](#performance-results)
8. [Real-World Applications](#real-world-applications)
9. [Notebooks](#notebooks)
10. [Configuration](#configuration)

---

## Overview

This system addresses the end-to-end lifecycle of a **speech trigger classification** task:

| Stage | What it does |
|-------|-------------|
| **Feature Extraction** | Extracts MFCCs, log-Mel spectrograms, and temporal descriptors (ZCR, RMS, spectral centroid, rolloff) from raw audio or simulated waveforms |
| **Data Generation** | Synthesises 1 M+ labelled samples with accent variations (neutral, British, Indian, Australian) and variable noise levels (10–40 dB SNR) |
| **Model Training** | Trains a Scikit-learn gradient-boosting model, a PyTorch DNN, and a **hybrid ensemble** that combines both |
| **Evaluation** | Reports accuracy, weighted F1, ROC-AUC (OVR), and generates confusion-matrix heatmaps & ROC curves |
| **Explainability** | Runs SHAP (Tree + Kernel), intrinsic & permutation feature importance, feature correlation, and per-accent performance analysis |

---

## Project Structure

```
Speech-Trigger-Analysis/
├── configs/
│   └── default.yaml              # Central YAML config
├── data/                         # Generated .npz splits (train/val/test)
├── experiments/
│   ├── checkpoints/              # Saved model weights
│   ├── results/                  # Evaluation reports & plots
│   ├── explainability/           # SHAP plots, importance reports
│   └── logs/                     # Run logs
├── notebooks/
│   ├── 01_feature_exploration.ipynb
│   ├── 02_training_demo.ipynb
│   └── 03_explainability_demo.ipynb
├── src/
│   ├── features/
│   │   ├── extractor.py          # MFCC, Mel-spectrogram, temporal features
│   │   └── audio_utils.py        # Normalisation, noise, accent perturbation
│   ├── models/
│   │   └── classifier.py         # Sklearn, DNN, Hybrid classifiers
│   ├── training/
│   │   ├── trainer.py            # Unified training loop
│   │   └── callbacks.py          # Early stopping, checkpointing
│   ├── evaluation/
│   │   ├── metrics.py            # Accuracy, F1, ROC-AUC, confusion matrix
│   │   └── evaluator.py          # Full eval pipeline + plots
│   ├── explainability/
│   │   ├── shap_explainer.py     # SHAP integration
│   │   ├── feature_importance.py # Intrinsic + permutation importance
│   │   └── accent_analysis.py    # Per-accent performance analysis
│   └── pipeline/
│       ├── data_generator.py     # Large-scale data simulation
│       ├── data_loader.py        # Batch loaders (NumPy / PyTorch)
│       └── pipeline.py           # End-to-end orchestrator + CLI
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline

```bash
# Generate 1M samples → train → evaluate → explain
python -m src.pipeline.pipeline --config configs/default.yaml

# Smaller demo run
python -m src.pipeline.pipeline --samples 10000

# Skip data generation (reuse existing .npz)
python -m src.pipeline.pipeline --skip-datagen
```

### 3. Explore notebooks

```bash
jupyter notebook notebooks/
```

---

## Feature Engineering Strategy

### MFCC (Mel-Frequency Cepstral Coefficients)
- 13 coefficients extracted via a pure-NumPy Mel filterbank → DCT pipeline.
- Summary statistics (mean, std, min, max) compress variable-length spectrograms into a **fixed 52-dim** vector per sample.

### Log-Mel Spectrogram
- 128-band Mel filterbank produces a detailed frequency representation.
- Same 4-statistic aggregation yields **512 features**.

### Temporal / Spectral Descriptors
| Feature | Description |
|---------|-------------|
| Zero-Crossing Rate | Rate of sign changes — distinguishes voiced vs. unvoiced |
| RMS Energy | Overall signal power |
| Spectral Centroid | "Centre of mass" of the spectrum — correlates with brightness |
| Spectral Rolloff | Frequency below which 85 % of energy is concentrated |

### Total feature vector: **568 dimensions** (52 MFCC + 512 Mel + 4 temporal).

### Multi-Condition Augmentation
- **Accents**: Neutral, British, Indian, Australian — simulated via formant-like spectral shifting.
- **Noise**: White Gaussian noise at 10–40 dB SNR to mimic real-world conditions.

---

## Model Architecture

### 1. Sklearn Classifier (Gradient Boosting)
- 200 estimators, max depth 6, learning rate 0.1.
- Provides **intrinsic feature importances** and is compatible with SHAP's fast `TreeExplainer`.

### 2. PyTorch DNN
```
Input (568) → Linear(256) → BN → ReLU → Dropout(0.3)
           → Linear(128) → BN → ReLU → Dropout(0.3)
           → Linear(64)  → BN → ReLU → Dropout(0.3)
           → Linear(4)   → Softmax
```
- Trained with Adam, cosine LR schedule, early stopping (patience 5).
- Higher capacity for capturing nonlinear feature interactions.

### 3. Hybrid Ensemble
- **Weighted probability averaging**: 40 % Sklearn + 60 % DNN.
- Combines the interpretability of tree models with the representational power of neural networks.

---

## Explainability Insights

### SHAP Analysis
- **TreeExplainer** (fast, exact) for the gradient-boosting component.
- **KernelExplainer** fallback for the DNN and hybrid models.
- Outputs: bar plot (mean |SHAP|), beeswarm summary, single-instance waterfall.

### Feature Importance
- **Intrinsic**: Tree-based `feature_importances_` from the sklearn model.
- **Permutation**: Model-agnostic — shuffles one feature at a time and measures accuracy drop.
- **Correlation analysis**: Identifies redundant features in the 568-dim space.

### Accent Variation Analysis
- Per-accent metrics (accuracy, F1, ROC-AUC) with side-by-side confusion matrices.
- Feature-distribution histograms stratified by accent.
- Highlights potential **bias**: if the model underperforms on certain accents, the reports surface it immediately.

Generated plots are saved to `experiments/explainability/`:

| File | Description |
|------|-------------|
| `shap_summary.png` | SHAP beeswarm plot |
| `shap_bar.png` | Top-20 features by mean |SHAP| |
| `shap_single_instance.png` | Waterfall for one prediction |
| `intrinsic_importance.png` | Tree-based feature ranking |
| `permutation_importance.png` | Permutation-based ranking |
| `feature_correlation.png` | Top-20 feature correlation heatmap |
| `accent_comparison.png` | Grouped bar chart of per-accent metrics |
| `accent_confusion_matrices.png` | Side-by-side CMs per accent |
| `accent_feature_distributions.png` | Feature histograms by accent |

---

## Performance Results

Results on synthetic data (10 000-sample demo; 1 M-sample runs yield tighter distributions):

| Model | Accuracy | F1 (weighted) | ROC-AUC (OVR) |
|-------|----------|---------------|----------------|
| Sklearn (GB) | ~0.72 | ~0.72 | ~0.90 |
| DNN | ~0.68 | ~0.68 | ~0.87 |
| **Hybrid** | **~0.74** | **~0.74** | **~0.91** |

> **Note**: These are indicative results on synthetic data with controlled class signatures. Real-world performance depends on dataset quality and domain-specific tuning.

Evaluation outputs are saved to `experiments/results/`:
- `evaluation_report.json` — machine-readable metric dump
- `confusion_matrix.png` — heatmap
- `roc_curves.png` — per-class ROC curves

---

## Real-World Applications

| Domain | Use Case |
|--------|----------|
| **Smart Speakers** | Wake-word detection with accent-robust trigger classification |
| **Automotive** | In-cabin voice command recognition under road noise |
| **Healthcare** | Monitoring vocal biomarkers (e.g., detecting stress triggers in speech) |
| **Security** | Speaker-aware command verification across diverse populations |
| **Call Centres** | Real-time trigger/keyword spotting for compliance monitoring |

The explainability layer is critical for:
- **Regulatory compliance** — SHAP reports provide per-prediction explanations.
- **Bias auditing** — accent analysis reveals performance disparities.
- **Feature selection** — importance rankings guide sensor / microphone design.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_feature_exploration.ipynb` | Visualise waveforms, MFCCs, Mel spectrograms, temporal features, accent effects |
| `02_training_demo.ipynb` | Full training workflow with loss curves and model comparison |
| `03_explainability_demo.ipynb` | SHAP, feature importance, accent analysis — end-to-end |

Launch with:
```bash
jupyter notebook notebooks/
```

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key sections:

```yaml
audio:
  sample_rate: 16000
  duration_sec: 2.0

features:
  mfcc: { n_mfcc: 13, n_fft: 2048, hop_length: 512 }
  mel_spectrogram: { n_mels: 128 }
  temporal: { zero_crossing: true, rms_energy: true, ... }

data:
  num_samples: 1_000_000
  accents: [neutral, british, indian, australian]
  noise_levels: [40, 30, 20, 10]

model:
  type: hybrid       # hybrid | sklearn | dnn
  sklearn: { estimator: gradient_boosting, n_estimators: 200 }
  dnn: { hidden_dims: [256, 128, 64], dropout: 0.3 }

training:
  epochs: 30
  learning_rate: 0.001
  early_stopping: { patience: 5 }

explainability:
  shap: { max_samples: 500 }
  feature_importance: { top_k: 20 }
  accent_analysis: { enabled: true }
```

Create custom experiment configs (e.g., `configs/high_noise.yaml`) by overriding individual sections.

---

## License

MIT
