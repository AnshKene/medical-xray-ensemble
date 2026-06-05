"""
run_gradcam.py — Grad-CAM visualisation for all 4 ensemble models.

Loads each trained model, runs Grad-CAM on test.png, and saves an overlay
PNG to outputs/gradcam/<model_name>_gradcam.png.

Run AFTER main.py has been executed (models must be trained and saved).
"""

import os
import torch
from PIL import Image

from main import get_resnet, get_mobilenet, get_efficientnet, get_densenet
from utils.gradcam import generate_gradcam
from utils.transforms import val_transform

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "test.png")
GRADCAM_DIR = os.path.join(BASE_DIR, "outputs", "gradcam")

# Model name → (factory function, saved weights path, target layer getter)
# target_layer_fn receives the loaded model and returns the convolutional layer
# that Grad-CAM should hook into.
MODEL_CONFIGS = [
    {
        "name":             "resnet",
        "factory":          get_resnet,
        "weights":          os.path.join(BASE_DIR, "outputs", "models", "resnet.pth"),
        "target_layer_fn":  lambda m: m.layer4,               # last ResNet block
    },
    {
        "name":             "mobilenet",
        "factory":          get_mobilenet,
        "weights":          os.path.join(BASE_DIR, "outputs", "models", "mobilenet.pth"),
        "target_layer_fn":  lambda m: m.features[-1],         # last MobileNetV3 block
    },
    {
        "name":             "efficientnet",
        "factory":          get_efficientnet,
        "weights":          os.path.join(BASE_DIR, "outputs", "models", "efficientnet.pth"),
        "target_layer_fn":  lambda m: m.features[-1],         # last EfficientNet block
    },
    {
        "name":             "densenet",
        "factory":          get_densenet,
        "weights":          os.path.join(BASE_DIR, "outputs", "models", "densenet.pth"),
        "target_layer_fn":  lambda m: m.features.denseblock4, # last DenseNet block
    },
]


def main():
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load and preprocess the input image ───────────────────────────────────
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(
            f"Input image not found: {IMAGE_PATH}\n"
            "Place a test X-ray image named test.png in the project root."
        )

    img_pil = Image.open(IMAGE_PATH).convert("RGB")
    img_tensor = val_transform(img_pil).unsqueeze(0).to(device)   # (1,3,224,224)
    print(f"Image loaded: {IMAGE_PATH}  shape={tuple(img_tensor.shape)}\n")

    # ── Run Grad-CAM for each model ───────────────────────────────────────────
    for cfg in MODEL_CONFIGS:
        name        = cfg["name"]
        weights_path = cfg["weights"]
        save_path   = os.path.join(GRADCAM_DIR, f"{name}_gradcam.png")

        print(f"[{name}] Processing...")

        # Check weights exist
        if not os.path.exists(weights_path):
            print(f"  ⚠  Weights not found: {weights_path} — skipping.\n"
                   "     Run main.py first to train and save all models.")
            continue

        # Build and load model
        model = cfg["factory"]().to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.eval()

        # Resolve target layer
        target_layer = cfg["target_layer_fn"](model)

        # Generate and save Grad-CAM
        try:
            pred_class = generate_gradcam(model, img_tensor, target_layer, save_path)
            print(f"  ✓  Predicted class index : {pred_class}")
            print(f"  ✓  Grad-CAM saved        : {save_path}\n")
        except Exception as e:
            print(f"  ✗  Grad-CAM failed for {name}: {e}\n")

    print("All done. Check outputs/gradcam/ for overlay images.")


if __name__ == "__main__":
    main()