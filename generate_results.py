import os
import json
import torch
from torch.utils.data import DataLoader

from utils.dataset import XRayDataset
from utils.transforms import val_transform
from utils.evaluate import evaluate

from main import (
    get_resnet,
    get_mobilenet,
    get_efficientnet,
    get_densenet,
)


def load_model(model_fn, path, device):
    model = model_fn().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    # -------------------------
    # 1. Load Dataset
    # -------------------------
    test_dataset = XRayDataset("data/test", val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    class_names = test_dataset.classes

    print("Classes:", class_names)

    # -------------------------
    # 2. Load Models
    # -------------------------
    model_paths = {
        "resnet":       "outputs/models/resnet.pth",
        "mobilenet":    "outputs/models/mobilenet.pth",
        "efficientnet": "outputs/models/efficientnet.pth",
        "densenet":     "outputs/models/densenet.pth",
    }

    print("\nLoading trained models...")

    m1 = load_model(get_resnet,       model_paths["resnet"],       device)
    m2 = load_model(get_mobilenet,    model_paths["mobilenet"],    device)
    m3 = load_model(get_efficientnet, model_paths["efficientnet"], device)
    m4 = load_model(get_densenet,     model_paths["densenet"],     device)

    models_list = [m1, m2, m3, m4]

    # -------------------------
    # 3. Ensemble Weights
    # -------------------------
    # Load weights saved by main.py, or fall back to equal weights
    weights_path = "outputs/metrics/ensemble_weights.json"
    if os.path.exists(weights_path):
        with open(weights_path) as f:
            w_dict = json.load(f)
        weights = [w_dict[n] for n in ["resnet", "mobilenet", "efficientnet", "densenet"]]
        print("\nLoaded ensemble weights:", weights)
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
        print("\nFallback equal weights (run main.py first to get computed weights)")

    # -------------------------
    # 4. Evaluate Ensemble
    # -------------------------
    print("\nRunning evaluation on test set...")

    report, preds = evaluate(
        models_list,
        weights,
        test_loader,
        device,
        class_names
    )

    print("\nEvaluation complete.")
    print("Check outputs/metrics for confusion matrix.")

    return report


if __name__ == "__main__":
    main()