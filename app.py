"""
app.py — Flask web server for the Ensemble X-Ray web interface.

Routes:
  GET  /          → serve the single-page UI
  POST /predict   → accept uploaded image, return prediction JSON
  GET  /status    → check if models are loaded and ready

Models are loaded lazily on the first /predict request and cached in memory.
"""

import os
import uuid
import json
import traceback
from io import BytesIO

import torch
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image

import inference as inf

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(__file__), "static", "gradcam"), exist_ok=True)

# ── Global model cache ────────────────────────────────────────────────────────
_models  = None
_weights = None
_names   = None
_missing = None
_device  = None
_load_error = None


def _ensure_models_loaded():
    """Lazily load models on first call; cache in module-level globals."""
    global _models, _weights, _names, _missing, _device, _load_error

    if _models is not None:
        return True, None

    if _load_error is not None:
        return False, _load_error

    try:
        _device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _models, _weights, _names, _missing = inf.load_all_models(_device)

        if not _models:
            _load_error = (
                "No trained models found in outputs/models/. "
                "Please run main.py first to train the ensemble."
            )
            return False, _load_error

        return True, None

    except Exception as e:
        _load_error = f"Failed to load models: {str(e)}"
        return False, _load_error


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/status")
def status():
    """Check whether models are loaded and ready."""
    ok, err = _ensure_models_loaded()
    if ok:
        return jsonify({
            "ready":   True,
            "device":  str(_device),
            "models":  _names,
            "missing": _missing,
        })
    return jsonify({"ready": False, "error": err}), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a multipart/form-data upload with key 'image'.
    Returns JSON with prediction, probabilities, model weights, and Grad-CAM URLs.
    """
    # ── Model readiness ───────────────────────────────────────────────────────
    ok, err = _ensure_models_loaded()
    if not ok:
        return jsonify({"error": err}), 503

    # ── Image validation ──────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use field name 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename — please select an image."}), 400

    allowed = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported file type '{ext}'. Use PNG or JPEG."}), 415

    # ── Load image ────────────────────────────────────────────────────────────
    try:
        img_bytes = file.read()
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Could not read image: {str(e)}"}), 422

    # ── Save upload (for display in the browser) ──────────────────────────────
    upload_id   = str(uuid.uuid4())[:8]
    upload_name = f"{upload_id}{ext}"
    upload_path = os.path.join(UPLOAD_FOLDER, upload_name)
    pil_image.save(upload_path)

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        result = inf.predict(_models, _weights, _names, _device, pil_image)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    try:
        gradcam_urls, gradcam_errors = inf.run_gradcam_all(
            _models, _names, _device, pil_image, prefix=upload_id
        )
    except Exception as e:
        traceback.print_exc()
        gradcam_urls   = {}
        gradcam_errors = {"all": str(e)}

    # ── Build response ────────────────────────────────────────────────────────
    response = {
        **result,
        "upload_url":     f"/static/uploads/{upload_name}",
        "gradcam_urls":   gradcam_urls,
        "gradcam_errors": gradcam_errors,
        "missing_models": _missing,
        "device":         str(_device),
    }
    return jsonify(response)


@app.route("/static/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  [X-Ray AI]  Ensemble X-Ray Web Interface")
    print("  ----------------------------------------")
    print("  Open: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
