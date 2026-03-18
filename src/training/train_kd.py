"""
src/training/train_kd.py
------------------------
Knowledge Distillation training loop: YOLOv8l teacher → YOLOv8s student.

Training strategy:
  - AdamW optimizer with separate LR for student and adapters
  - CosineAnnealingLR scheduler
  - Mixed precision (AMP) via torch.amp.GradScaler
  - Validation every epoch using YOLO's native val pipeline
  - Early stopping on mAP50
  - Checkpointing at configurable intervals
"""

import gc
import shutil
import yaml
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from ultralytics import YOLO

from src.models.architectures import FrozenTeacher, Student, build_feature_adapters
from src.training.kd_loss import KDLoss

log = logging.getLogger("KD")


def validate_student(
    student: Student,
    yaml_path: str,
    master_classes: list,
    nc: int,
    img_size: int,
    batch_size: int,
    output_dir: Path,
    device
) -> tuple:
    """
    Run YOLO native validation on the current student weights.

    Returns:
        (mAP50, mAP50-95)
    """
    try:
        tmp_ckpt = output_dir / "tmp_student_val.pt"
        torch.save(student.model.state_dict(), tmp_ckpt)

        val_model = Student(nc=nc).to(device)
        val_model.model.load_state_dict(
            torch.load(tmp_ckpt, map_location='cpu'), strict=False
        )
        val_model.model.eval()
        val_model.model.nc = nc
        val_model.model.names = {i: master_classes[i] for i in range(nc)}

        y = YOLO('yolov8s.pt')
        y.model = val_model.model

        results = y.val(
            data=str(yaml_path),
            imgsz=img_size,
            batch=batch_size,
            device=0 if torch.cuda.is_available() else 'cpu',
            verbose=False,
            plots=False
        )

        map50    = float(results.box.map50) if hasattr(results, 'box') else 0.0
        map5095  = float(results.box.map)   if hasattr(results, 'box') else 0.0

        if tmp_ckpt.exists():
            tmp_ckpt.unlink()

        return map50, map5095

    except Exception as e:
        log.warning(f"Validation error: {type(e).__name__}: {str(e)[:80]}")
        return 0.0, 0.0


def train_kd(
    yaml_path: str,
    teacher_weights: str,
    student_weights: str,
    master_classes: list,
    train_loader,
    val_loader,
    output_dir: str,
    nc: int = 19,
    img_size: int = 640,
    batch_size: int = 16,
    epochs: int = 120,
    patience: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    temperature: float = 3.0,
    alpha: float = 0.5,
    beta: float = 0.3,
    gamma: float = 0.2,
    save_every: int = 10,
    check_every: int = 10,
    seed: int = 42
) -> Path:
    """
    Run the full KD training loop.

    Args:
        yaml_path: Path to YOLO data.yaml.
        teacher_weights: Path to pretrained teacher .pt file.
        student_weights: Base student weights (e.g. 'yolov8s.pt' or pretrained path).
        master_classes: Ordered list of class names.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (currently unused; YOLO val used instead).
        output_dir: Directory for checkpoints and logs.
        nc: Number of classes.
        img_size: Input image size.
        batch_size: Batch size for validation.
        epochs: Max training epochs.
        patience: Early stopping patience (epochs without mAP50 improvement).
        lr: Student learning rate.
        weight_decay: AdamW weight decay.
        temperature: KL distillation temperature.
        alpha: Detection loss weight.
        beta: Feature loss weight.
        gamma: KL loss weight.
        save_every: Save checkpoint every N epochs.
        check_every: Run full validation every N epochs.
        seed: Random seed.

    Returns:
        Path to best student weights.
    """
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("🎓 KNOWLEDGE DISTILLATION TRAINING")
    print(f"   Teacher: {teacher_weights}")
    print(f"   Student: {student_weights}")
    print(f"   NC: {nc} | Epochs: {epochs} | T: {temperature}")
    print(f"   α={alpha} β={beta} γ={gamma}")
    print("=" * 70)

    # --- Models ---
    teacher = FrozenTeacher(teacher_weights).to(device)
    student = Student(nc=nc).to(device)

    # Load pretrained student backbone if available
    if Path(student_weights).exists() and not student_weights.endswith('.yaml'):
        y = YOLO(student_weights)
        student.model.load_state_dict(y.model.state_dict(), strict=False)
        log.info(f"Student initialized from: {student_weights}")

    adapters = build_feature_adapters(student, teacher, device)

    # --- Loss ---
    kd_loss = KDLoss(alpha=alpha, beta=beta, gamma=gamma, temperature=temperature, nc=nc)
    kd_loss.set_criterion(student.model)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW([
        {'params': student.parameters(),  'lr': lr,      'weight_decay': weight_decay},
        {'params': adapters.parameters(), 'lr': lr * 2,  'weight_decay': 0},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    # --- Training State ---
    best_map50 = 0.0
    best_epoch = 0
    no_improve  = 0
    best_ckpt   = output / 'best_kd_student.pt'

    for epoch in range(1, epochs + 1):
        student.train()
        adapters.train()
        teacher.eval()

        epoch_loss = 0.0
        det_sum = feat_sum = kl_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for imgs, targets in pbar:
            imgs    = imgs.to(device).float()
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                t_out   = teacher(imgs)
                t_feats = teacher.features()

                s_out   = student(imgs)
                s_feats = student.features()

                total, d, f, k = kd_loss(s_out, t_out, s_feats, t_feats, adapters, targets)

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(total)
            det_sum  += d
            feat_sum += f
            kl_sum   += k
            n_batches += 1

            pbar.set_postfix(loss=f"{float(total):.4f}", det=f"{d:.3f}", feat=f"{f:.3f}")

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        log.info(
            f"Epoch {epoch:3d} | loss={avg_loss:.4f} "
            f"det={det_sum/n_batches:.3f} feat={feat_sum/n_batches:.3f} kl={kl_sum/n_batches:.3f}"
        )

        # --- Periodic Validation ---
        if epoch % check_every == 0 or epoch == epochs:
            map50, map5095 = validate_student(
                student, yaml_path, master_classes, nc, img_size, batch_size, output, device
            )
            print(f"  [Epoch {epoch}] mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
            log.info(f"Epoch {epoch} | mAP50={map50:.4f} mAP50-95={map5095:.4f}")

            if map50 > best_map50:
                best_map50 = map50
                best_epoch = epoch
                no_improve  = 0
                torch.save({'epoch': epoch, 'model': student.model.state_dict(),
                            'map50': map50, 'map5095': map5095}, best_ckpt)
                print(f"  💾 New best saved: mAP50={map50:.4f}")
            else:
                no_improve += check_every
                if no_improve >= patience:
                    print(f"\n⏹  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                    break

        # --- Periodic Checkpoint ---
        if epoch % save_every == 0:
            ckpt_path = output / f'student_epoch{epoch:03d}.pt'
            torch.save({'epoch': epoch, 'model': student.model.state_dict()}, ckpt_path)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✅ Training complete. Best mAP50={best_map50:.4f} at epoch {best_epoch}")
    print(f"   Best weights: {best_ckpt}")
    return best_ckpt


if __name__ == "__main__":
    import argparse
    from src.data.dataset import build_dataloaders

    parser = argparse.ArgumentParser(description="KD Training: YOLOv8l → YOLOv8s")
    parser.add_argument("--yaml",     required=True)
    parser.add_argument("--teacher",  required=True)
    parser.add_argument("--student",  default="yolov8s.pt")
    parser.add_argument("--output",   default="outputs")
    parser.add_argument("--nc",       type=int, default=19)
    parser.add_argument("--epochs",   type=int, default=120)
    parser.add_argument("--batch",    type=int, default=16)
    args = parser.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)
    classes = list(cfg['names'].values()) if isinstance(cfg['names'], dict) else cfg['names']

    train_loader, val_loader = build_dataloaders(
        merged_dir=str(Path(args.yaml).parent),
        nc=args.nc,
        batch_size=args.batch
    )

    train_kd(
        yaml_path=args.yaml,
        teacher_weights=args.teacher,
        student_weights=args.student,
        master_classes=classes,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output,
        nc=args.nc,
        epochs=args.epochs,
        batch_size=args.batch
    )
