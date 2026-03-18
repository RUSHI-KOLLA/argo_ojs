"""
src/data/dataset.py
-------------------
PyTorch Dataset and DataLoader for YOLO-format weed detection data.
Includes augmentation pipelines tuned for Knowledge Distillation training
(no ImageNet normalization — pixel values scaled to [0, 1]).
"""

from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ----------------------------------------------------------------
# Augmentation Transforms
# ----------------------------------------------------------------

def get_transforms(img_size: int = 640, train: bool = True) -> A.Compose:
    """
    Build albumentations transform pipeline for YOLO KD training.

    IMPORTANT: Uses [0, 1] pixel scaling instead of ImageNet mean/std
    normalization. This is required for YOLO-style models.

    Args:
        img_size: Target image size (square).
        train: If True, includes geometric + color augmentations.

    Returns:
        Albumentations Compose pipeline with YOLO bbox params.
    """
    bbox_params = A.BboxParams(
        format='yolo',
        label_fields=['cls'],
        min_visibility=0.0,
        check_each_transform=False
    )

    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
            A.Affine(scale=(0.9, 1.1), translate_percent=(0.0, 0.05), rotate=(-15, 15), p=0.25),
            # Scale to [0, 1] only — no ImageNet normalization
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=bbox_params)


# ----------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------

class WeedDataset(Dataset):
    """
    YOLO-format weed detection dataset.

    Expects:
        img_dir/  *.jpg / *.png
        lbl_dir/  *.txt  (one line per box: class cx cy w h)

    Returns:
        (image_tensor [C,H,W], targets [N, 6])
        where targets columns are: [batch_idx, class, cx, cy, w, h]
        batch_idx is always 0 here; the collate_fn fills it in.
    """

    def __init__(self, img_dir: str, lbl_dir: str, transform: A.Compose):
        self.img_dir = Path(img_dir)
        self.lbl_dir = Path(lbl_dir)
        self.transform = transform

        exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
        imgs = []
        for e in exts:
            imgs += list(self.img_dir.glob(e))

        # Strict image-label pairing
        self.samples: List[Tuple[Path, Path]] = []
        for img in sorted(imgs):
            lbl = self.lbl_dir / (img.stem + '.txt')
            if lbl.exists():
                self.samples.append((img, lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load labels
        boxes, classes = [], []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                # Clip to valid range
                cx = max(0.001, min(0.999, cx))
                cy = max(0.001, min(0.999, cy))
                w  = max(0.001, min(0.999, w))
                h  = max(0.001, min(0.999, h))
                boxes.append([cx, cy, w, h])
                classes.append(cls)

        # Apply transforms
        if boxes:
            result = self.transform(image=image, bboxes=boxes, cls=classes)
            image = result['image']
            boxes = result['bboxes']
            classes = result['cls']
        else:
            result = self.transform(image=image, bboxes=[], cls=[])
            image = result['image']

        # Build target tensor [N, 6]: batch_idx=0, cls, cx, cy, w, h
        if boxes:
            target = torch.zeros((len(boxes), 6))
            target[:, 1] = torch.tensor(classes, dtype=torch.float32)
            target[:, 2:] = torch.tensor(boxes, dtype=torch.float32)
        else:
            target = torch.zeros((0, 6))

        return image.float(), target


# ----------------------------------------------------------------
# Collate Function
# ----------------------------------------------------------------

def collate_fn(batch):
    """
    Custom collate: fills batch_idx column (col 0) of each target
    with its position in the batch. Required for YOLO loss functions.
    """
    images, targets = zip(*batch)
    images = torch.stack(images)

    for i, tgt in enumerate(targets):
        if tgt.shape[0] > 0:
            tgt[:, 0] = i

    targets = torch.cat(targets, dim=0)
    return images, targets


# ----------------------------------------------------------------
# DataLoader Factory
# ----------------------------------------------------------------

def build_dataloaders(
    merged_dir: str,
    img_size: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    nc: int = 19,
    use_weighted_sampler: bool = True,
    seed: int = 42
):
    """
    Build train and validation DataLoaders with optional weighted sampling
    to handle class imbalance.

    Args:
        merged_dir: Root of merged YOLO dataset.
        img_size: Input image size.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        nc: Number of classes (for weight computation).
        use_weighted_sampler: Balance classes via WeightedRandomSampler.
        seed: Random seed.

    Returns:
        (train_loader, val_loader)
    """
    merged = Path(merged_dir)

    train_tf = get_transforms(img_size, train=True)
    val_tf   = get_transforms(img_size, train=False)

    train_ds = WeedDataset(
        merged / 'train' / 'images',
        merged / 'train' / 'labels',
        train_tf
    )
    val_ds = WeedDataset(
        merged / 'val' / 'images',
        merged / 'val' / 'labels',
        val_tf
    )

    sampler = None
    if use_weighted_sampler:
        sampler = _build_weighted_sampler(train_ds, nc)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda wid: np.random.seed(seed + wid)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader


def _build_weighted_sampler(dataset: WeedDataset, nc: int) -> WeightedRandomSampler:
    """Compute per-sample weights inversely proportional to class frequency."""
    class_counts: Counter = Counter()
    sample_classes = []

    for _, target in dataset:
        if target.shape[0] > 0:
            classes = target[:, 1].long().tolist()
            class_counts.update(classes)
            sample_classes.append(classes[0])  # Use first class as representative
        else:
            sample_classes.append(0)

    total = sum(class_counts.values()) or 1
    class_weight = {c: total / (nc * max(count, 1)) for c, count in class_counts.items()}

    sample_weights = [class_weight.get(c, 1.0) for c in sample_classes]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
