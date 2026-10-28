"""
Penn-Fudan Pedestrian Segmentation Dataset Loader (Compact Binary Tensor Stream)
Loaded directly from pre-packaged uint8 binary tensor (only 23MB, zero loose files).
Binary mask (0: background, 1: pedestrian).
"""

import os
import torch
from torch.utils.data import Dataset


class PennFudanDataset(Dataset):
    def __init__(self, root="./data", task="segmentation", split="train", img_size=128):
        self.split = split
        self.task = task
        self.img_size = img_size

        # Map test/val interchangeably
        clean_split = "val" if split in ["test", "val"] else "train"

        # Locate pennfudan_<split>.pt
        candidates = [
            os.path.join(root, f"pennfudan_{clean_split}.pt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/pennfudan_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/pennfudan_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/pennfudan_{clean_split}.pt")),
        ]
        pt_path = None
        for c in candidates:
            if os.path.exists(c):
                pt_path = c
                break

        if pt_path is None:
            raise RuntimeError(f"pennfudan_{split}.pt not found under {root}. Checked: {candidates}")

        data = torch.load(pt_path)
        self.images = data["images"]  # (N, 3, 128, 128) uint8
        self.masks = data["masks"]    # (N, 128, 128) long

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        img = (img - self.mean) / self.std
        return img, self.masks[idx]
