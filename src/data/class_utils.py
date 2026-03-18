"""
src/data/class_utils.py
-----------------------
Class discovery, name cleaning, and master class list construction
for the merged Cotton-Weed + MH-Weed16 dataset.
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple


def clean_name(name: str) -> str:
    """
    Normalize a raw class folder/label name into a clean title-cased string.
    Strips leading digits, parenthetical suffixes, underscores, and extra spaces.
    """
    name = re.sub(r'^\d+\.+', '', name).strip()
    if '(' in name:
        name = name.split('(')[0].strip()
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip().rstrip('.')
    return name.title()


def clean_class_name(name: str) -> str:
    """
    Secondary cleaner: strips trailing underscores and normalizes spacing.
    Used specifically for MH-Weed folder-derived names.
    """
    name = name.strip().rstrip('_')
    name = name.replace('_', ' ')
    name = ' '.join(word.capitalize() for word in name.split())
    return name


def load_cotton_classes(cotton_path: str) -> List[str]:
    """
    Load class names from the Cotton-Weed dataset's data.yaml.

    Args:
        cotton_path: Root path of the cotton-weed dataset.

    Returns:
        List of class name strings.
    """
    cotton_yaml_path = Path(cotton_path) / "data.yaml"
    assert cotton_yaml_path.exists(), f"Missing cotton data.yaml: {cotton_yaml_path}"

    with open(cotton_yaml_path, "r") as f:
        c_yaml = yaml.safe_load(f)

    classes = c_yaml["names"]
    if isinstance(classes, dict):
        classes = [classes[i] for i in range(len(classes))]
    return [str(x).strip() for x in classes]


def load_mhweed_classes(mhweed_path: str) -> List[str]:
    """
    Discover MH-Weed16 class names from folder structure.
    Resolves both known and auto-discovered class folder paths.

    Args:
        mhweed_path: Root of the MH-Weed16 dataset.

    Returns:
        List of cleaned class name strings (Sicklepod renamed to 'Sicklepod Mh').
    """
    mh_root = Path(mhweed_path)

    # Try known path first
    class_folder = mh_root / "Individual Weed Species" / "16 Classes of Weed_Species" / "Individual Weed_Species"

    # Auto-search fallback
    if not class_folder.exists():
        for p in mh_root.rglob('*'):
            if p.is_dir() and 'Weed_Species' in p.name:
                subdirs = [d for d in p.iterdir() if d.is_dir()]
                if len(subdirs) >= 10:
                    class_folder = p
                    break

    assert class_folder.exists(), f"MH class folder not found under: {mhweed_path}"

    classes = []
    for d in sorted(class_folder.iterdir()):
        if d.is_dir():
            n = clean_name(d.name)
            if n:
                classes.append(n)

    # Avoid collision with Cotton's Sicklepod
    classes = ["Sicklepod Mh" if c == "Sicklepod" else c for c in classes]
    return classes


def build_master_class_list(
    cotton_path: str,
    mhweed_path: str,
    expected_nc: int = 28
) -> Tuple[List[str], Dict[int, int], Dict[int, int]]:
    """
    Build unified master class list from Cotton-Weed + MH-Weed16.
    Deduplicates while preserving insertion order.

    Args:
        cotton_path: Root of cotton-weed dataset.
        mhweed_path: Root of MH-Weed16 dataset.
        expected_nc: Expected total number of unique classes.

    Returns:
        Tuple of (master_list, cotton_remap, mhweed_remap)
        where remap dicts map original_class_id → master_class_id.
    """
    cotton_classes = load_cotton_classes(cotton_path)
    mhweed_classes = load_mhweed_classes(mhweed_path)

    # Apply secondary cleaning to MH-Weed names
    mhweed_classes = [clean_class_name(c) for c in mhweed_classes if c.strip()]

    master = list(dict.fromkeys(
        cotton_classes + [c for c in mhweed_classes if c not in cotton_classes]
    ))

    nc = len(master)
    assert all(c.strip() for c in master), "Empty class name found in master list"
    assert nc == expected_nc, f"Expected {expected_nc} classes, got {nc}"

    cotton_remap = {i: master.index(c) for i, c in enumerate(cotton_classes)}
    mhweed_remap = {i: master.index(c) for i, c in enumerate(mhweed_classes) if c in master}

    return master, cotton_remap, mhweed_remap
