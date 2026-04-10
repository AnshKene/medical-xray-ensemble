import torch
import os
from torch.utils.data import DataLoader
from utils.dataset import XRayDataset
from utils.transforms import val_transform
from utils.evaluate import evaluate
from main import get_resnet, get_mobilenet, get_efficientnet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading saved models on: {device}")

    # 1. Load the Test Data
    test_dataset = XRayDataset("data/test", val_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    class_names = test_dataset.classes

    # 2. Initialize and Load Trained Weights
    # These must match your saved filenames exactly
    model_paths = {
        "resnet": "outputs/models/resnet.pth",
        "mobilenet": "outputs/models/mobilenet.pth",
        "efficientnet": "outputs/models/efficientnet.pth"
    }

    m1 = get_resnet().to(device)
    m1.load_state_dict(torch.load(model_paths["resnet"], weights_only=True))
    
    m2 = get_mobilenet().to(device)
    m2.load_state_dict(torch.load(model_paths["mobilenet"], weights_only=True))
    
    m3 = get_efficientnet().to(device)
    m3.load_state_dict(torch.load(model_paths["efficientnet"], weights_only=True))

    models_list = [m1, m2, m3]

    # 3. Use the weights from your successful run
    # (Based on your last output: ResNet=0.336, MobileNet=0.331, EfficientNet=0.333)
    weights = [0.336, 0.331, 0.333]

    # 4. Generate the Visual Metrics
    print("\nGenerating Confusion Matrix and Final Report...")
    report, preds = evaluate(models_list, weights, test_loader, device, class_names)

    print("\nSUCCESS: Check 'outputs/metrics/confusion_matrix.png' for the visual results.")

if __name__ == "__main__":
    main()