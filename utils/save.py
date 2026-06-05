import os
import torch
import json
import csv
from datetime import datetime


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)


def save_model(model, name):
    """Save model state_dict. Fixed name (no timestamp) so generate_results.py can load it."""
    path = f"outputs/models/{name}.pth"
    ensure_dir(path)
    torch.save(model.state_dict(), path)
    print(f"  Saved model → {path}")


def save_metrics(metrics, name):
    """Save classification report dict as timestamped JSON."""
    timestamp = get_timestamp()
    path = f"outputs/metrics/{name}_{timestamp}.json"
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"  Saved metrics → {path}")


def save_predictions(preds, name, class_names=None):
    """
    Save predictions as a timestamped CSV.
    If class_names is provided, includes a human-readable label column.
    """
    timestamp = get_timestamp()
    path = f"outputs/predictions/{name}_{timestamp}.csv"
    ensure_dir(path)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if class_names:
            writer.writerow(["Index", "Predicted_Class_Index", "Predicted_Class_Name"])
            for idx, p in enumerate(preds):
                label = class_names[p] if 0 <= p < len(class_names) else str(p)
                writer.writerow([idx, p, label])
        else:
            writer.writerow(["Index", "Predicted_Class_Index"])
            for idx, p in enumerate(preds):
                writer.writerow([idx, p])

    print(f"  Saved predictions → {path}")