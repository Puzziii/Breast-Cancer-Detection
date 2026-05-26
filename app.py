import gradio as gr
import torch
import torch.nn as nn
import joblib
import numpy as np

# ── Model definition (must match train_and_save.py) ───────────────────────────
class BreastCancerNN(nn.Module):
    def __init__(self, input_dim=30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ── Load artifacts ─────────────────────────────────────────────────────────────
model = BreastCancerNN()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

scaler  = joblib.load("scaler.pkl")
classes = joblib.load("classes.pkl")   # ['B', 'M']

# ── Feature names (from Wisconsin dataset) ────────────────────────────────────
FEATURES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# ── Prediction function ────────────────────────────────────────────────────────
def predict(*args):
    features = np.array(args, dtype=np.float32).reshape(1, -1)
    features_scaled = scaler.transform(features)
    tensor = torch.tensor(features_scaled, dtype=torch.float32)

    with torch.no_grad():
        prob = model(tensor).item()

    label = "Malignant 🔴" if prob > 0.5 else "Benign 🟢"
    confidence = prob if prob > 0.5 else 1 - prob

    return {
        label: float(confidence),
        ("Benign 🟢" if prob > 0.5 else "Malignant 🔴"): float(1 - confidence)
    }

# ── Default example values (mean of dataset) ─────────────────────────────────
DEFAULTS = [
    14.13, 20.38, 91.96, 654.9, 0.1173, 0.1277, 0.0865, 0.0497,
    0.1812, 0.0667, 0.4063, 1.216, 2.833, 40.79, 0.0065, 0.0218,
    0.0257, 0.0097, 0.0184, 0.0064, 16.27, 25.68, 107.3, 827.8,
    0.1468, 0.2377, 0.2671, 0.1015, 0.2475, 0.0895
]

# ── Build Gradio UI ────────────────────────────────────────────────────────────
inputs = [
    gr.Number(label=feat, value=val)
    for feat, val in zip(FEATURES, DEFAULTS)
]

demo = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=gr.Label(label="Prediction", num_top_classes=2),
    title="🩺 Breast Cancer Detection",
    description=(
        "Enter cell nucleus measurements from a fine needle aspirate (FNA) biopsy. "
        "The model predicts whether the mass is **Benign** or **Malignant**.\n\n"
        "> Built with PyTorch · Wisconsin Breast Cancer Dataset · by [Poojitha](https://github.com/Puzziii)"
    ),
    examples=[[*DEFAULTS]],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
