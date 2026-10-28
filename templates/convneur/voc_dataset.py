"""
Pascal VOC 2007 Object Detection Dataset Loader (Compact Binary Tensor Stream)
Loaded directly from pre-packaged uint8 binary tensor (only 47MB, zero loose files).
20 classes:
  aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow,
  diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
"""

import os
import torch
from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(VOC_CLASSES)}


class VOCDetection(Dataset):
    def __init__(self, root="./data", split="train", img_size=128, max_samples=None):
        self.split = split
        self.img_size = img_size

        # Map test/val interchangeably
        clean_split = "val" if split in ["test", "val"] else "train"

        # Locate voc_detection_<split>.pt
        candidates = [
            os.path.join(root, f"voc_detection_{clean_split}.pt"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../../data/voc_detection_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"../data/voc_detection_{clean_split}.pt")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), f"data/voc_detection_{clean_split}.pt")),
        ]
        pt_path = None
        for c in candidates:
            if os.path.exists(c):
                pt_path = c
                break

        if pt_path is None:
            raise RuntimeError(f"voc_detection_{split}.pt not found under {root}. Checked: {candidates}")

        data = torch.load(pt_path)
        self.images = data["images"]  # (N, 3, 128, 128) uint8
        self.boxes = data["boxes"]    # (N, 4) float32 [cx, cy, w, h]
        self.labels = data["labels"]  # (N,) long [0..19]

        if max_samples and max_samples < len(self.images):
            self.images = self.images[:max_samples]
            self.boxes = self.boxes[:max_samples]
            self.labels = self.labels[:max_samples]

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].float() / 255.0
        img = (img - self.mean) / self.std
        return img, self.boxes[idx], self.labels[idx]
