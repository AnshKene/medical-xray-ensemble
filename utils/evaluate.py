import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def get_model_weight(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    return accuracy_score(y_true, y_pred)

def weighted_predict(models, weights, images, device):
    probs = []
    images = images.to(device)
    for model, w in zip(models, weights):
        model.eval()
        with torch.no_grad():
            outputs = model(images)
            p = F.softmax(outputs, dim=1)
            probs.append(w * p)
            del outputs
            torch.cuda.empty_cache()
    final = torch.sum(torch.stack(probs), dim=0)
    return torch.argmax(final, dim=1)

def evaluate(models, weights, loader, device, class_names):
    y_true, y_pred = [], []

    print("Running Ensemble Inference...")
    for images, labels in loader:
        preds = weighted_predict(models, weights, images, device)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    # 1. Classification Report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    print("\n--- Final Ensemble Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # 2. Confusion Matrix Plotting
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Ensemble Confusion Matrix')
    
    # Save the plot
    plt.savefig('outputs/metrics/confusion_matrix.png')
    print("Confusion Matrix saved to outputs/metrics/confusion_matrix.png")
    plt.close()

    return report, y_pred