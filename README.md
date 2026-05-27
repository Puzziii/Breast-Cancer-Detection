# Breast Cancer Detection

Binary classifier for malignant/benign breast mass prediction — built from scratch in PyTorch, no sklearn classifiers, no pretrained weights.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/puzziii/breast-cancer-detection)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20BC-green)](https://github.com/gscdit/Breast-Cancer-Detection)

---

## Overview

Takes 30 cell nucleus measurements from a breast mass fine needle aspirate (FNA) → outputs **Malignant** or **Benign** with a confidence score.

The model is a single-layer neural network written entirely in raw PyTorch — no Trainer APIs, no pretrained weights, no sklearn classifiers. Every forward pass, loss calculation, and gradient update is written by hand.

---

## Architecture

```
Input (30 features)
    ↓
nn.Linear(30 → 1)
    ↓
Sigmoid activation
    ↓
Binary prediction + confidence
```

| Hyperparameter | Value |
|---|---|
| Loss | Binary Cross Entropy |
| Optimizer | SGD |
| Learning rate | 0.1 |
| Epochs | 25 |
| Preprocessing | StandardScaler + LabelEncoder |

---

## Dataset

[Wisconsin Breast Cancer Dataset](https://github.com/gscdit/Breast-Cancer-Detection) — 569 samples, 30 real-valued features computed from digitized images of FNA biopsies. Features describe characteristics of cell nuclei: radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

Labels: `M` (Malignant) · `B` (Benign)

---

## Project Structure

```
breast-cancer-detection/
├── app.py                 # Gradio UI and inference logic
├── train_and_save.py      # Full training pipeline
├── model.pt               # Saved model weights
├── scaler.pkl             # Fitted StandardScaler
├── classes.pkl            # LabelEncoder class mapping
└── requirements.txt
```

---

## Run Locally

```bash
git clone https://github.com/your-username/breast-cancer-detection
cd breast-cancer-detection
pip install -r requirements.txt
```

Train the model:

```bash
python train_and_save.py
```

Launch the Gradio app:

```bash
python app.py
```

The app runs at `http://localhost:7860`.

---

## Stack

| Layer | Tool |
|---|---|
| Model | PyTorch `nn.Module` |
| UI | Gradio |
| Deployment | Hugging Face Spaces |
| Training | Google Colab |
| Dataset | Wisconsin Breast Cancer |

---

## Notes

This is a learning project — purpose is to understand what happens under the hood of a binary classifier. The architecture is intentionally minimal: one linear layer, one activation, raw gradient updates.

For production medical use, a much more robust validation process, larger dataset, and calibrated uncertainty estimates would be required.
