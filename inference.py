"""
inference.py — Reusable inference engine for the Ensemble X-Ray web interface.

Responsibilities:
  • Load all 4 trained models from outputs/models/
  • Run weighted softmax ensemble prediction on a PIL image
  • Generate Grad-CAM heatmaps for all 4 models

Designed to be imported by app.py (Flask) or used standalone.
"""

import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from main import get_resnet, get_mobilenet, get_efficientnet, get_densenet
from utils.transforms import val_transform
from utils.gradcam import generate_gradcam

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(BASE_DIR, "outputs", "models")
WEIGHTS_JSON = os.path.join(BASE_DIR, "outputs", "metrics", "ensemble_weights.json")
GRADCAM_DIR  = os.path.join(BASE_DIR, "static", "gradcam")

CLASS_NAMES = ["covid", "normal", "pneumonia"]

# Model name → (factory function, target layer getter)
MODEL_CONFIGS = [
    {
        "name":            "resnet",
        "factory":         get_resnet,
        "weights_file":    "resnet.pth",
        "target_layer_fn": lambda m: m.layer4,
    },
    {
        "name":            "mobilenet",
        "factory":         get_mobilenet,
        "weights_file":    "mobilenet.pth",
        "target_layer_fn": lambda m: m.features[-1],
    },
    {
        "name":            "efficientnet",
        "factory":         get_efficientnet,
        "weights_file":    "efficientnet.pth",
        "target_layer_fn": lambda m: m.features[-1],
    },
    {
        "name":            "densenet",
        "factory":         get_densenet,
        "weights_file":    "densenet.pth",
        "target_layer_fn": lambda m: m.features.denseblock4,
    },
]


# ── Model loader ──────────────────────────────────────────────────────────────

def load_all_models(device: torch.device):
    """
    Load all 4 trained models and ensemble weights from disk.

    Returns:
        models  : list of nn.Module (eval mode)
        weights : list of float (sum ≈ 1.0)
        names   : list of str
        missing : list of str  (models whose .pth file was not found)
    """
    os.makedirs(GRADCAM_DIR, exist_ok=True)

    models, weights, names, missing = [], [], [], []

    # Load saved ensemble weights (fall back to uniform if file absent)
    if os.path.exists(WEIGHTS_JSON):
        with open(WEIGHTS_JSON) as f:
            saved_weights = json.load(f)
    else:
        saved_weights = {cfg["name"]: 0.25 for cfg in MODEL_CONFIGS}

    for cfg in MODEL_CONFIGS:
        path = os.path.join(MODELS_DIR, cfg["weights_file"])
        if not os.path.exists(path):
            missing.append(cfg["name"])
            continue

        model = cfg["factory"]().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        models.append(model)
        weights.append(saved_weights.get(cfg["name"], 0.25))
        names.append(cfg["name"])

    # Re-normalise weights in case some models are missing
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    return models, weights, names, missing


# ── Single-image prediction ───────────────────────────────────────────────────

def predict(models, weights, names, device: torch.device, pil_image: Image.Image):
    """
    Run weighted ensemble inference on a single PIL image.

    Returns a dict:
        prediction    : str   — winning class name
        confidence    : float — probability of winning class (0–1)
        probabilities : dict  — {class_name: float, ...}
        model_weights : dict  — {model_name: float, ...}
    """
    img_tensor = val_transform(pil_image).unsqueeze(0).to(device)  # (1,3,224,224)

    fused = torch.zeros(1, len(CLASS_NAMES), device=device)
    for model, w in zip(models, weights):
        with torch.no_grad():
            p = F.softmax(model(img_tensor), dim=1)
        fused += w * p

    probs      = fused.squeeze(0).cpu().numpy()          # shape: (num_classes,)
    pred_idx   = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "prediction":    pred_class,
        "confidence":    round(confidence, 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, probs)},
        "model_weights": {n: round(w, 4) for n, w in zip(names, weights)},
    }


# ── Grad-CAM for all models ───────────────────────────────────────────────────

def run_gradcam_all(models, names, device: torch.device,
                    pil_image: Image.Image, prefix: str = "upload"):
    """
    Generate Grad-CAM heatmaps for each loaded model.

    Args:
        prefix : filename prefix (e.g. the upload UUID) to avoid cache collisions

    Returns:
        gradcam_urls : dict  — {model_name: '/static/gradcam/<filename>.png'}
        errors       : dict  — {model_name: error_message}  (models that failed)
    """
    os.makedirs(GRADCAM_DIR, exist_ok=True)
    img_tensor = val_transform(pil_image).unsqueeze(0).to(device)

    gradcam_urls = {}
    errors       = {}

    for model, name in zip(models, names):
        cfg = next(c for c in MODEL_CONFIGS if c["name"] == name)
        target_layer = cfg["target_layer_fn"](model)
        filename     = f"{prefix}_{name}_gradcam.png"
        save_path    = os.path.join(GRADCAM_DIR, filename)

        try:
            generate_gradcam(model, img_tensor, target_layer, save_path)
            gradcam_urls[name] = f"/static/gradcam/{filename}"
        except Exception as e:
            errors[name] = str(e)

    return gradcam_urls, errors
