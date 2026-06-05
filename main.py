import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split

from torchvision.models import (
    resnet18,           ResNet18_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    efficientnet_b0,    EfficientNet_B0_Weights,
    densenet121,        DenseNet121_Weights,
)

from utils.dataset   import XRayDataset
from utils.transforms import train_transform, val_transform
from utils.bagging   import create_bootstrap
from utils.train     import train_model
from utils.evaluate  import get_model_weight, evaluate
from utils.save      import save_model, save_metrics, save_predictions

NUM_CLASSES = 3

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True   # fully deterministic CUDA ops
    torch.backends.cudnn.benchmark     = False  # disable auto-tuning for reproducibility


# ── Directory creation ─────────────────────────────────────────────────────────
def create_dirs():
    for d in ["outputs/models", "outputs/logs",
              "outputs/metrics", "outputs/predictions", "outputs/gradcam"]:
        os.makedirs(d, exist_ok=True)


# ── Helper: freeze all backbone parameters ─────────────────────────────────────
def freeze(model):
    for p in model.parameters():
        p.requires_grad = False


# ── Model factory functions ────────────────────────────────────────────────────

def get_resnet():
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    freeze(m)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def get_mobilenet():
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    freeze(m)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    return m


def get_efficientnet():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    freeze(m)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m


def get_densenet():
    m = densenet121(weights=DenseNet121_Weights.DEFAULT)
    freeze(m)
    m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
    return m


# VGG-16 removed: 138M params, no skip connections — overfits on medical datasets
# with minimal diversity gain over the 4 remaining architectures.


# ── Main pipeline ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    create_dirs()

    # ── Datasets ───────────────────────────────────────────────────────────────
    train_dataset = XRayDataset("data/train", train_transform)
    val_dataset   = XRayDataset("data/val",   val_transform)
    test_dataset  = XRayDataset("data/test",  val_transform)

    class_names = test_dataset.classes
    print(f"Classes  : {class_names}")
    print(f"Train    : {len(train_dataset)} images")
    print(f"Val      : {len(val_dataset)} images")
    print(f"Test     : {len(test_dataset)} images\n")

    # ── Split val → val_model (training signal) + val_ensemble (weight estimation)
    # This avoids data leakage when computing ensemble weights.
    n_ens   = max(1, int(0.3 * len(val_dataset)))
    n_model = len(val_dataset) - n_ens
    g = torch.Generator().manual_seed(SEED)
    val_model_ds, val_ens_ds = random_split(val_dataset, [n_model, n_ens], generator=g)

    # ── DataLoaders ────────────────────────────────────────────────────────────
    # num_workers=0  → safest on Windows (avoids multiprocessing spawn issues)
    # pin_memory=True → faster CPU→GPU transfer when CUDA is available
    pin = torch.cuda.is_available()

    val_loader     = DataLoader(val_model_ds, batch_size=16, shuffle=False,
                                 num_workers=0, pin_memory=pin)
    val_ens_loader = DataLoader(val_ens_ds,   batch_size=16, shuffle=False,
                                 num_workers=0, pin_memory=pin)
    test_loader    = DataLoader(test_dataset,  batch_size=16, shuffle=False,
                                 num_workers=0, pin_memory=pin)

    # ── Bootstrap bags (one per model) ────────────────────────────────────────
    loaders = [
        DataLoader(create_bootstrap(train_dataset), batch_size=16,
                   shuffle=True, num_workers=0, pin_memory=pin)
        for _ in range(4)
    ]

    log_file = "outputs/logs/training_log.txt"

    # ── Train all 4 models ────────────────────────────────────────────────────
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    models = [
        train_model(get_resnet(),       loaders[0], val_loader, device, log_file=log_file),
        train_model(get_mobilenet(),    loaders[1], val_loader, device, log_file=log_file),
        train_model(get_efficientnet(), loaders[2], val_loader, device, log_file=log_file),
        train_model(get_densenet(),     loaders[3], val_loader, device, log_file=log_file),
    ]

    names = ["resnet", "mobilenet", "efficientnet", "densenet"]

    # ── Save trained models ────────────────────────────────────────────────────
    print("\nSaving models...")
    for m, n in zip(models, names):
        save_model(m, n)

    # ── Ensemble weights (computed on held-out val_ens, not the training val) ──
    print("\nComputing ensemble weights on held-out val split...")
    raw_weights = [get_model_weight(m, val_ens_loader, device) for m in models]
    total       = sum(raw_weights)
    weights     = [w / total for w in raw_weights]

    for n, w, rw in zip(names, weights, raw_weights):
        print(f"  {n:<14} raw_acc={rw:.4f}  weight={w:.4f}")

    # ── Evaluate ensemble on test set ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    report, preds = evaluate(models, weights, test_loader, device, class_names)

    # ── Persist results ────────────────────────────────────────────────────────
    save_metrics(report, "results")
    save_predictions(preds, "predictions", class_names=class_names)

    weights_dict = dict(zip(names, weights))
    with open("outputs/metrics/ensemble_weights.json", "w") as f:
        json.dump(weights_dict, f, indent=2)
    print("  Ensemble weights saved → outputs/metrics/ensemble_weights.json")

    print("\n✓ DONE")


if __name__ == "__main__":
    main()