# utils/dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset

VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

class XRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        
        # Ignore hidden folders/files when getting classes
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(VALID_EXTENSIONS):
                    self.data.append((os.path.join(cls_path, img_name), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label