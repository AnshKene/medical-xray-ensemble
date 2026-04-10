import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision.models import (
    resnet18, ResNet18_Weights, 
    mobilenet_v3_small, MobileNet_V3_Small_Weights, 
    efficientnet_b0, EfficientNet_B0_Weights
)

from utils.dataset import XRayDataset
from utils.transforms import train_transform, val_transform
from utils.bagging import create_bootstrap
from utils.train import train_model
from utils.evaluate import get_model_weight, evaluate
from utils.save import save_model, save_metrics, save_predictions

# --------------------
# 1. Global Setup
# --------------------
NUM_CLASSES = 3

def create_dirs():
    for d in ["outputs/models", "outputs/logs", "outputs/metrics", "outputs/predictions"]:
        os.makedirs(d, exist_ok=True)

def get_resnet():
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m

def get_mobilenet():
    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    return m

def get_efficientnet():
    m = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m

# --------------------
# 2. Main Execution Block
# --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on: {device}")

    create_dirs()

    print("\nLoading datasets...")
    train_dataset = XRayDataset("data/train", train_transform)
    val_dataset   = XRayDataset("data/val", val_transform)
    test_dataset  = XRayDataset("data/test", val_transform)

    # Note: num_workers=4 will now work perfectly on Windows
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("\nBootstrapping training datasets...")
    boot1 = create_bootstrap(train_dataset)
    boot2 = create_bootstrap(train_dataset)
    boot3 = create_bootstrap(train_dataset)

    loader1 = DataLoader(boot1, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    loader2 = DataLoader(boot2, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    loader3 = DataLoader(boot3, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

    log_file = "outputs/logs/training_log.txt"

    print("\n--- Training Model 1: ResNet18 ---")
    model1 = train_model(get_resnet(), loader1, val_loader, device, epochs=10, log_file=log_file)
    save_model(model1, "resnet")

    print("\n--- Training Model 2: MobileNetV3 ---")
    model2 = train_model(get_mobilenet(), loader2, val_loader, device, epochs=10, log_file=log_file)
    save_model(model2, "mobilenet")

    print("\n--- Training Model 3: EfficientNetB0 ---")
    model3 = train_model(get_efficientnet(), loader3, val_loader, device, epochs=10, log_file=log_file)
    save_model(model3, "efficientnet")

    models_list = [model1, model2, model3]

    print("\nCalculating ensemble weights based on validation performance...")
    w1 = get_model_weight(model1, val_loader, device)
    w2 = get_model_weight(model2, val_loader, device)
    w3 = get_model_weight(model3, val_loader, device)

    total = w1 + w2 + w3
    weights = [w1/total, w2/total, w3/total]

    print(f"Final Ensemble Weights: ResNet={weights[0]:.3f}, MobileNet={weights[1]:.3f}, EfficientNet={weights[2]:.3f}")

    print("\nRunning final evaluation on isolated test set...")
    report, preds = evaluate(models_list, weights, test_loader, device)

    save_metrics(report, "results")
    save_predictions(preds, "predictions")

    print("\nPipeline complete. Results saved in outputs/")


# --------------------
# 3. The Windows Multiprocessing Shield
# --------------------
if __name__ == '__main__':
    main()