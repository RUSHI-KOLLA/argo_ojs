"""
src/data/cleaning.py
--------------------
Post-merge data cleaning pipeline:
  1. Remove fake/full-image bounding boxes
  2. Merge rare classes into OtherWeed
  3. Remove empty classes from label files + YAML
  4. Oversample underrepresented classes
"""

import shutil
import random
import yaml
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


# ----------------------------------------------------------------
# 1. Fake Box Removal
# ----------------------------------------------------------------

def remove_fake_boxes(
    merged_dir: str,
    area_threshold: float = 0.90,
    wh_threshold: float = 0.95
) -> Dict[str, int]:
    """
    Remove bounding boxes that cover nearly the entire image (likely annotation errors).
    If no valid boxes remain in a file, deletes both the label and its image.

    Args:
        merged_dir: Path to merged dataset root.
        area_threshold: Remove boxes where w*h > this value.
        wh_threshold: Remove boxes where both w > this AND h > this.

    Returns:
        Summary dict with counts of removed boxes and deleted images per split.
    """
    merged = Path(merged_dir)
    summary = {}

    for split in ['train', 'val', 'test']:
        img_dir = merged / split / 'images'
        lbl_dir = merged / split / 'labels'
        if not lbl_dir.exists():
            continue

        removed, deleted = 0, 0

        for lbl_file in list(lbl_dir.glob('*.txt')):
            txt = lbl_file.read_text().strip()
            if not txt:
                continue

            new_lines = []
            found_fake = False

            for line in txt.splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                if w * h > area_threshold or (w > wh_threshold and h > wh_threshold):
                    removed += 1
                    found_fake = True
                else:
                    new_lines.append(line.strip())

            if found_fake:
                if not new_lines:
                    lbl_file.unlink()
                    for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                        img_f = img_dir / f"{lbl_file.stem}{ext}"
                        if img_f.exists():
                            img_f.unlink()
                            deleted += 1
                            break
                else:
                    lbl_file.write_text('\n'.join(new_lines))

        summary[split] = {'removed_boxes': removed, 'deleted_images': deleted}
        print(f"  {split}: {removed} fake boxes removed, {deleted} images deleted")

    return summary


# ----------------------------------------------------------------
# 2. Rare Class Merging
# ----------------------------------------------------------------

def merge_rare_classes(
    merged_dir: str,
    yaml_path: str,
    rare_threshold: int = 400,
    merge_target_name: str = "OtherWeed"
) -> str:
    """
    Merge classes with fewer than `rare_threshold` samples into OtherWeed.
    Updates all label files and the data.yaml.

    Args:
        merged_dir: Root of merged dataset.
        yaml_path: Path to data.yaml.
        rare_threshold: Classes below this count get merged.
        merge_target_name: Name of the catch-all class.

    Returns:
        Path to updated data.yaml.
    """
    merged = Path(merged_dir)

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg['names']
    nc = data_cfg['nc']

    # Count class instances in train + val
    class_counts: Counter = Counter()
    for split in ['train', 'val']:
        for txt in (merged / split / 'labels').glob('*.txt'):
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    print("\nClass distribution:")
    rare_classes = []
    for cls_id in range(nc):
        count = class_counts.get(cls_id, 0)
        status = "✅ Keep" if count >= rare_threshold else "🔴 MERGE"
        print(f"  Class {cls_id:2d} ({names.get(cls_id, '?'):25s}): {count:5d}  [{status}]")
        if count < rare_threshold:
            rare_classes.append(cls_id)

    # Find merge target index
    merge_target_id = next(
        (k for k, v in names.items() if v == merge_target_name),
        rare_classes[0] if rare_classes else 0
    )

    # Build old→new remap
    old_to_new: Dict[int, int] = {}
    for i in range(nc):
        if i in rare_classes:
            old_to_new[i] = merge_target_id
        else:
            shift = sum(1 for r in rare_classes if r < i and r != merge_target_id)
            old_to_new[i] = i - shift

    print(f"\nMerging {rare_classes} → Class {merge_target_id} ({merge_target_name})")
    _remap_all_labels(merged, old_to_new)

    # Update YAML
    new_names = {}
    new_id = 0
    for old_id in range(nc):
        if old_id in rare_classes and old_id != merge_target_id:
            continue
        new_names[new_id] = merge_target_name if old_id == merge_target_id else names[old_id]
        new_id += 1

    data_cfg['nc'] = len(new_names)
    data_cfg['names'] = new_names
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)

    print(f"  Updated YAML: {nc} → {len(new_names)} classes")
    return yaml_path


# ----------------------------------------------------------------
# 3. Empty Class Removal
# ----------------------------------------------------------------

def remove_empty_classes(merged_dir: str, yaml_path: str) -> str:
    """
    Remove classes that have zero annotated instances, updating
    all label files and the data.yaml accordingly.
    """
    merged = Path(merged_dir)

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg['names']
    nc = data_cfg['nc']

    class_counts: Counter = Counter()
    for split in ['train', 'val', 'test']:
        for txt in (merged / split / 'labels').glob('*.txt'):
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    class_counts[int(parts[0])] += 1

    empty = [i for i in range(nc) if class_counts.get(i, 0) == 0]
    if not empty:
        print("  No empty classes found.")
        return yaml_path

    print(f"  Removing empty classes: {[names.get(c, c) for c in empty]}")

    remap: Dict[int, int] = {}
    new_names: Dict[int, str] = {}
    new_id = 0
    for old_id in range(nc):
        if old_id in empty:
            remap[old_id] = -1
        else:
            remap[old_id] = new_id
            new_names[new_id] = names[old_id]
            new_id += 1

    # Remap labels, delete any that become empty
    for split in ['train', 'val', 'test']:
        lbl_dir = merged / split / 'labels'
        img_dir = merged / split / 'images'
        if not lbl_dir.exists():
            continue
        for lbl_file in list(lbl_dir.glob('*.txt')):
            new_lines = []
            for line in lbl_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                old_cls = int(float(parts[0]))
                if remap.get(old_cls, -1) == -1:
                    continue
                new_lines.append(f"{remap[old_cls]} {' '.join(parts[1:])}")
            if not new_lines:
                lbl_file.unlink()
                for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                    img_f = img_dir / f"{lbl_file.stem}{ext}"
                    if img_f.exists():
                        img_f.unlink()
                        break
            else:
                lbl_file.write_text('\n'.join(new_lines))

    data_cfg['nc'] = len(new_names)
    data_cfg['names'] = new_names
    with open(yaml_path, 'w') as f:
        yaml.dump(data_cfg, f)

    print(f"  Updated YAML: {nc} → {len(new_names)} classes")
    return yaml_path


# ----------------------------------------------------------------
# 4. Oversampling
# ----------------------------------------------------------------

def oversample_classes(
    merged_dir: str,
    yaml_path: str,
    target_count: int = 750,
    seed: int = 42
) -> None:
    """
    Duplicate images of underrepresented classes to reach `target_count`.
    Only operates on the training split.

    Args:
        merged_dir: Root of merged dataset.
        yaml_path: Path to data.yaml.
        target_count: Minimum desired samples per class.
        seed: Random seed.
    """
    random.seed(seed)
    merged = Path(merged_dir)

    with open(yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    nc = data_cfg['nc']

    train_img_dir = merged / 'train' / 'images'
    train_lbl_dir = merged / 'train' / 'labels'

    # Map class → list of label files containing that class
    class_to_files: Dict[int, List[Path]] = {i: [] for i in range(nc)}
    for lbl_file in sorted(train_lbl_dir.glob('*.txt')):
        seen = set()
        for line in lbl_file.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls not in seen:
                    class_to_files[cls].append(lbl_file)
                    seen.add(cls)

    print("\nOversampling underrepresented classes:")
    for cls_id in range(nc):
        count = len(class_to_files[cls_id])
        if count == 0 or count >= target_count:
            continue

        needed = target_count - count
        sources = random.choices(class_to_files[cls_id], k=needed)
        print(f"  Class {cls_id}: {count} → {count + needed}")

        for i, src_lbl in enumerate(sources):
            suffix = f"_os{i:04d}"
            new_stem = src_lbl.stem + suffix

            # Copy label
            new_lbl = train_lbl_dir / (new_stem + '.txt')
            shutil.copy2(src_lbl, new_lbl)

            # Copy image
            for ext in ['.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG']:
                src_img = train_img_dir / (src_lbl.stem + ext)
                if src_img.exists():
                    shutil.copy2(src_img, train_img_dir / (new_stem + ext))
                    break


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _remap_all_labels(merged: Path, old_to_new: Dict[int, int]) -> None:
    """Remap class IDs across all splits in-place."""
    for split in ['train', 'val', 'test']:
        lbl_dir = merged / split / 'labels'
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.glob('*.txt'):
            lines = []
            for line in txt.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                new_cls = old_to_new.get(cls, cls)
                lines.append(f"{new_cls} {' '.join(parts[1:])}")
            txt.write_text('\n'.join(lines))
