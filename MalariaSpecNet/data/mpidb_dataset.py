# MP-IDB three-class dataset (falciparum, vivax, ovale) with 7-channel transform

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A



def to7ch(img_bgr: np.ndarray) -> np.ndarray:

    # RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # L from LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, _, _ = cv2.split(lab)

    # S from HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, S, _ = cv2.split(hsv)

    # Laplacian gradient (abs normalized to 0-255)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    grad = cv2.Laplacian(gray, cv2.CV_32F)
    grad = cv2.normalize(np.abs(grad), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Morphological top-hat on gray
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    chs = [img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2], L, S, grad, tophat]
    out = np.stack(chs, axis=-1).astype(np.float32) / 255.0
    return out


def _build_transforms(img_size=100, aug=True):
    if aug:
        return A.Compose([
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15,
                               border_mode=cv2.BORDER_REFLECT101, p=0.9),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.GaussNoise(var_limit=(5, 15), p=0.2),
            A.Resize(img_size, img_size),
        ])
    else:
        return A.Compose([A.Resize(img_size, img_size)])


class MpidbDataset(Dataset):

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        root: str = "data/MPIDB",
        split: str = "train",                     # 'train'|'val'|'test'
        classes: list = ("falciparum", "vivax", "ovale", "negative"),
        img_size: int = 100,
        aug: bool = True, 
    ):
        super().__init__()
        split = split.lower()
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"

        self.root = root
        self.split = split
        self.classes = [c.lower() for c in classes]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.tf = _build_transforms(img_size=img_size, aug=(aug and split == "train"))
        self.paths, self.labels = self._scan()

        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in {root}/{split} for classes {self.classes}")

    def _scan(self):
        paths, labels = [], []
        base = os.path.join(self.root, self.split)
        for cls in self.classes:
            d = os.path.join(base, cls)
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith(self.IMG_EXTS):
                    paths.append(os.path.join(d, f))
                    labels.append(self.class_to_idx[cls])
        return paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        y = self.labels[idx]
        img_bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")

        arr7 = to7ch(img_bgr)                # (H, W, 7), float32 [0,1]
        arr7 = self.tf(image=arr7)["image"]  # Albumentations 期望 (H, W, C)
        chw = np.transpose(arr7, (2, 0, 1))  # -> (7, H, W)

        x = torch.from_numpy(chw).float()
        y = torch.tensor(y, dtype=torch.long)
        return x, y
