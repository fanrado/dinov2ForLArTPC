import os, zlib, numpy as np
from glob import glob
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
import sys

class CustomImagesDataset(Dataset):
    """
    ViT transformer expect 3 channels RGB-like, the input should fit this scheme
    Returns (image, dummy_target). DINOv2 will apply its own multi-crop transform.
    """
    def __init__(self, root,
                 split=None,
                 swap_axes=False,
                 transform=None,
                 target_transform=None,
                 **kwargs):
        self.root = root
        self.split = split
        self.swap_axes = bool(swap_axes)
        self.transform = transform
        self.target_transform = target_transform

        self.entries = []
        if not os.path.isdir(root):
            raise RuntimeError(f"{root} is not a directory!")
        for gz in sorted(glob(os.path.join(root, "*.JPEG"))):
            key = os.path.splitext(os.path.basename(gz))[0]
            self.entries.append((key, gz))

        if not self.entries:
            raise RuntimeError(f"No .JPEG files found under {root}")

    def __len__(self):
        return len(self.entries)

    def _read_array(self, gz_path):
        with open(gz_path, "rb") as f:
            arr = BytesIO(f.read())
        return arr  # (3, H, W)

    def _to_pil(self, arr3):
        return Image.open(arr3).convert(mode="RGB")

    def __getitem__(self, idx):
        _, gz = self.entries[idx]
        arr = self._read_array(gz)
        img = self._to_pil(arr)
        target = 0  # dummy label for SSL
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


