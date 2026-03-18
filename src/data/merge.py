"""
src/data/merge.py
-----------------
Merges Cotton-Weed and MH-Weed16 datasets into a single unified
YOLO-format dataset with consistent class IDs.

Pipeline:
  1. Reset merged directory
  2. Copy + remap Cotton-Weed labels/images
  3. Copy + remap MH-Weed16 labels/images (YOLO_darknet format)
  4. Write unified data.yaml
"""

import shutil
import random
import yaml
from pathlib import Path
from typing import Dict, List, Optional

from .class_utils import build_master_class_list


def valid_img_list(img_dir: Path) -> List[Path]:
    """Return all valid image files in a directory."""
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    out = []
    for e in exts:
        out.extend(img_dir.glob(e))
    return out


def remap_label_lines(lines: List[str], remap_dict: Dict[int, int]) -> List[str]:
    """
    Remap class IDs in YOLO label lines.
    Skips lines with invalid format or unmapped class IDs.
    Validates bounding box coordinates are in [0, 1].
    """
    mapped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_cls = int(float(parts[0]))
        if old_cls not in remap_dict:
            continue
        x, y, w, h = map(float, parts[1:])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            continue  # Skip boxes with out-of-range coordinates
        mapped.append(f"{remap_dict[old_cls]} {x} {y} {w} {h}")
    return mapped


def merge_datasets(
    cotton_path: str,
    mhweed_path: str,
    merged_dir: str,
    master_classes: List[str],
    cotton_remap: Dict[int, int],
    mhweed_remap: Dict[int, int],
    val_split: float = 0.15,
    test_split: float = 0.05,
    seed: int = 42
) -> Path:
    """
    Merge Cotton-Weed and MH-Weed16 into a single YOLO dataset.

    Args:
        cotton_path: Root path of cotton-weed dataset.
        mhweed_path: Root of MH-Weed16 dataset.
        merged_dir: Output directory for merged dataset.
        master_classes: Unified class name list.
        cotton_remap: cotton original_id → master_id mapping.
        mhweed_remap: mhweed original_id → master_id mapping.
        val_split: Fraction of data for validation.
        test_split: Fraction of data for test.
        seed: Random seed for reproducibility.

    Returns:
        Path to generated data.yaml.
    """
    random.seed(seed)

    merged = Path(merged_dir)
    cotton = Path(cotton_path) / 'cotton_weed'

    data_root = Path(mhweed_path)
    yolo_labels = (
        data_root / 'Crop with Weeds'
        / 'intel Real Sense Depth_Annotations'
        / 'intel Real Sense Depth_Annotations'
        / 'YOLO_darknet'
    )
    mhweed_img_dirs = [
        data_root / 'Crop with Weeds' / 'Canon Camera_Clicks' / 'Canon Camera_Clicks',
        data_root / 'Crop with Weeds' / 'iPhone_Clicks' / 'iPhone_Clicks',
        data_root / 'Crop with Weeds' / 'intel Real Sense Depth_Clicks' / 'intel Real Sense Depth_Clicks',
    ]

    # Reset merged directory
    if merged.exists():
        shutil.rmtree(merged)
    for split in ['train', 'val', 'test']:
        (merged / split / 'images').mkdir(parents=True, exist_ok=True)
        (merged / split / 'labels').mkdir(parents=True, exist_ok=True)

    print(f"Cotton: {cotton} (exists: {cotton.exists()})")
    print(f"YOLO labels: {len(list(yolo_labels.glob('*.txt')))} files")

    # --- Cotton-Weed ---
    _merge_cotton(cotton, merged, cotton_remap, val_split, test_split, seed)

    # --- MH-Weed16 ---
    _merge_mhweed(mhweed_img_dirs, yolo_labels, merged, mhweed_remap, val_split, test_split, seed)

    # Write data.yaml
    yaml_path = merged / 'data.yaml'
    data_cfg = {
        'path': str(merged),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(master_classes),
        'names': {i: n for i, n in enumerate(master_classes)},
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f, default_flow_style=False)

    print(f"\n✅ Merged dataset written to: {merged}")
    print(f"   data.yaml: {yaml_path}")
    return yaml_path


def _merge_cotton(cotton_dir, merged, remap, val_split, test_split, seed):
    """Internal: copy and remap cotton-weed split data."""
    for split in ['train', 'val', 'test']:
        img_dir = cotton_dir / split / 'images'
        lbl_dir = cotton_dir / split / 'labels'
        if not img_dir.exists():
            continue
        for img in valid_img_list(img_dir):
            lbl = lbl_dir / (img.stem + '.txt')
            if not lbl.exists():
                continue
            lines = lbl.read_text().splitlines()
            remapped = remap_label_lines(lines, remap)
            if not remapped:
                continue
            dst_img = merged / split / 'images' / img.name
            dst_lbl = merged / split / 'labels' / (img.stem + '.txt')
            shutil.copy2(img, dst_img)
            dst_lbl.write_text('\n'.join(remapped))


def _merge_mhweed(img_dirs, label_dir, merged, remap, val_split, test_split, seed):
    """Internal: copy and remap MH-Weed16 data with random splits."""
    random.seed(seed)
    label_files = list(label_dir.glob('*.txt'))

    for lbl_file in label_files:
        lines = lbl_file.read_text().splitlines()
        remapped = remap_label_lines(lines, remap)
        if not remapped:
            continue

        # Find matching image
        img = None
        for img_dir in img_dirs:
            for ext in ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG']:
                candidate = img_dir / (lbl_file.stem + ext)
                if candidate.exists():
                    img = candidate
                    break
            if img:
                break
        if img is None:
            continue

        r = random.random()
        if r < test_split:
            split = 'test'
        elif r < test_split + val_split:
            split = 'val'
        else:
            split = 'train'

        shutil.copy2(img, merged / split / 'images' / img.name)
        (merged / split / 'labels' / (img.stem + '.txt')).write_text('\n'.join(remapped))
