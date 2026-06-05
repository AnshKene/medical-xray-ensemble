# utils/transforms.py
from torchvision import transforms

# ImageNet statistics — required by all pretrained backbones
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD  = [0.229, 0.224, 0.225]

# ── Training transform ────────────────────────────────────────────────────────
# Medical X-ray specific augmentation:
#   • ColorJitter  — simulates different X-ray exposure levels
#   • GaussianBlur — simulates slight motion / detector noise
#   • RandomVerticalFlip (p=0.1) — handles PA vs AP projections
#   • NO RandomResizedCrop — cutting anatomy is harmful for diagnosis
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.1),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])

# ── Validation / Test transform ───────────────────────────────────────────────
# Deterministic — no augmentation, same pipeline as training normalization
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
])