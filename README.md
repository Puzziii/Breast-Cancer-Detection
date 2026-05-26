# 🩺 Breast Cancer Detection - PyTorch from Scratch
Binary classifier to detect breast cancer (Malignant/Benign) built from scratch using PyTorch. Trained on the Wisconsin Breast Cancer Dataset. Deployed on Hugging Face Spaces with Gradio.

> No `sklearn` classifiers. No pretrained weights. Just tensors, math, and a sigmoid.

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow)](https://huggingface.co/spaces/puzziii/breast-cancer-detection)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20BC-green)](https://github.com/gscdit/Breast-Cancer-Detection)

---

## What it does

takes 30 cell nucleus measurements from a breast mass biopsy → predicts **Malignant** or **Benign** with confidence score.

built a neural network completely from scratch using raw PyTorch — no high-level trainer APIs, no pretrained anything. every forward pass, loss calculation, and gradient update written by hand.

---

## how it works

```
Input (30 features) → Linear layer → Sigmoid → Binary prediction
```

- **Loss**: Binary Cross Entropy
- **Optimizer**: SGD (lr=0.1)
- **Epochs**: 25
- **Preprocessing**: StandardScaler + LabelEncoder

---

## stack

| what | tool |
|------|------|
| Model | PyTorch (nn.Module) |
| UI | Gradio |
| Deployment | Hugging Face Spaces |
| Dataset | Wisconsin Breast Cancer |
| Training | Google Colab |

---

## project structure

```
📦 breast-cancer-detection
├── app.py                 # gradio interface
├── train_and_save.py      # training pipeline
├── model.pt               # saved weights
├── scaler.pkl             # fitted StandardScaler
├── classes.pkl            # label encoder classes
└── requirements.txt
```

---

## run locally

```bash
git clone https://github.com/Puzziii/breast-cancer-detection
cd breast-cancer-detection
pip install -r requirements.txt
python app.py
```
