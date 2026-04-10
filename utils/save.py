# utils/save.py
import os
import torch
import json
import csv

def ensure_dir(filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

def save_model(model, name):
    path = f"outputs/models/{name}.pth"
    ensure_dir(path)
    torch.save(model.state_dict(), path)

def save_metrics(metrics, name):
    path = f"outputs/metrics/{name}.json"
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

def save_predictions(preds, name):
    path = f"outputs/predictions/{name}.csv"
    ensure_dir(path)
    # Save as CSV for easy pandas loading later
    with open(path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Predicted_Class"])
        for idx, p in enumerate(preds):
            writer.writerow([idx, p])