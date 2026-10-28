"""
CIFAR-10 Dataset Loader (Cached HuggingFace / PyTorch Dataset)
50,000 train images, 10,000 test images (10 classes).
Image size: 32x32x3.
"""

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CIFAR10HF(Dataset):
    def __init__(self, split="train", transform=None):
        from datasets import load_dataset
        # Load directly from local Hugging Face cache (offline-safe)
        self.ds = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["img"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        label = int(item["label"])

        if self.transform is not None:
            img = self.transform(img)

        return img, label
