import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def get_model_weight(model, loader, device):
    """Return per-model validation accuracy (used as ensemble weight)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images  = images.to(device)
            outputs = model(images)
            preds   = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred)


def _get_ensemble_scores(models, weights, images, device):
    """
    Weighted softmax fusion.
    Returns a (batch, num_classes) tensor of fused probabilities.
    ✅ empty_cache is called ONCE per batch, not inside the model loop.
    """
    probs  = []
    images = images.to(device)
    for model, w in zip(models, weights):
        model.eval()
        with torch.no_grad():
            p = F.softmax(model(images), dim=1)
            probs.append(w * p)
    torch.cuda.empty_cache()          # ✅ once per batch — safe and fast
    return torch.sum(torch.stack(probs), dim=0)


def evaluate(models, weights, loader, device, class_names):
    """
    Run weighted ensemble inference on a DataLoader.
    Produces:
      • classification report (dict)
      • AUC-ROC (macro, one-vs-rest)
      • confusion matrix PNG saved to outputs/metrics/
    Returns (report_dict, y_pred_list).
    """
    y_true, y_pred, y_scores = [], [], []

    print("Running Ensemble Inference...")
    for images, labels in loader:
        scores = _get_ensemble_scores(models, weights, images, device)
        preds  = torch.argmax(scores, dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(scores.cpu().numpy())

    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    y_scores = np.array(y_scores)

    # 1. Classification Report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    print("\n--- Final Ensemble Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # 2. AUC-ROC (critical for medical AI)
    try:
        auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
        print(f"Macro AUC-ROC : {auc:.4f}")
        report["macro_auc_roc"] = float(auc)
    except Exception as e:
        print(f"AUC-ROC skipped: {e}")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/metrics/confusion_matrix.png", dpi=150)
    print("Confusion Matrix saved → outputs/metrics/confusion_matrix.png")
    plt.close()

    return report, y_pred.tolist()