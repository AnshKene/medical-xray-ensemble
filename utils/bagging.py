# utils/bagging.py
import torch
from torch.utils.data import Subset

def create_bootstrap(dataset):
    dataset_size = len(dataset)
    # Vectorized random sampling with replacement
    indices = torch.randint(0, dataset_size, (dataset_size,)).tolist()
    return Subset(dataset, indices)