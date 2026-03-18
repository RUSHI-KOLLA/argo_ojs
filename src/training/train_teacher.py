"""
src/training/train_teacher.py
------------------------------
Train the YOLOv8l teacher model on the merged 19-class weed dataset.
Uses Ultralytics native training API with strong augmentation settings.

Expected result: mAP50 ≈ 0.79, mAP50-95 ≈ 0.60
"""

import shutil
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO


def train_teacher(
    yaml_path: str,
    output_dir: str,
    epochs: int = 150,
    patience: int = 40,
    batch: int = 8,
    img_size: int = 640,
    device: int = 0,
    seed: int = 42
) -> Path:
    """
    Train the YOLOv8l teacher model.

    Args:
        yaml_path: Path to YOLO data.yaml.
        output_dir: Directory to save training outputs.
        epochs: Max training epochs.
        patience: Early stopping patience.
        batch: Batch size (use 8 for YOLOv8l to fit in GPU memory).
        img_size: Input image size.
        device: GPU device index.
        seed: Random seed.

    Returns:
        Path to best teacher weights (.pt).
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎓 TRAINING TEACHER — YOLOv8l (19-class weed detection)")
    print("=" * 70)

    model = YOLO('yolov8l.pt')

    results = model.train(
        data=str(yaml_path),

        # Schedule
        epochs=epochs,
        patience=patience,

        # Input
        imgsz=img_size,
        batch=batch,

        # Hardware
        cache='disk',
        workers=4,
        amp=True,
        device=device,

        # LR
        lr0=0.01,
        lrf=0.01,
        warmup_epochs=5,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,

        # Optimizer
        optimizer='AdamW',
        weight_decay=0.0005,

        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        fliplr=0.5,
        flipud=0.0,
        close_mosaic=15,

        # Loss
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # Output
        project=str(output),
        name='teacher_train',
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=5,
        val=True,
        seed=seed,
        deterministic=False,
    )

    # Copy best weights to output root for easy access
    best_src = output / 'teacher_train' / 'weights' / 'best.pt'
    best_dst = output / 'best_teacher.pt'
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\n✅ Teacher weights saved: {best_dst}")

    try:
        print(f"   mAP50:    {float(results.box.map50):.4f}")
        print(f"   mAP50-95: {float(results.box.map):.4f}")
    except Exception:
        print("   Check outputs/teacher_train/results.csv for metrics")

    return best_dst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLOv8l teacher model")
    parser.add_argument("--yaml",   required=True, help="Path to data.yaml")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch",  type=int, default=8)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    train_teacher(
        yaml_path=args.yaml,
        output_dir=args.output,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device
    )
