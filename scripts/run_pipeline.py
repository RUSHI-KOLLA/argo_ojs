"""
scripts/run_pipeline.py
-----------------------
End-to-end pipeline runner for argo_ojs weed detection project.

Stages:
  1. Build master class list from both datasets
  2. Merge datasets into unified YOLO format
  3. Clean data (fake boxes, rare class merging, empty class removal)
  4. Oversample underrepresented classes
  5. Train teacher (YOLOv8l)   [optional: skip if weights provided]
  6. Train student via KD      [YOLOv8s]
  7. Quantize + evaluate

Usage:
  # Full pipeline from scratch
  python scripts/run_pipeline.py --config configs/config.yaml

  # Skip teacher training (use existing weights)
  python scripts/run_pipeline.py --config configs/config.yaml --teacher-weights outputs/best_teacher.pt

  # KD only (teacher + student weights already exist)
  python scripts/run_pipeline.py --config configs/config.yaml \\
    --teacher-weights outputs/best_teacher.pt \\
    --skip-merge --skip-teacher
"""

import argparse
import logging
import sys
import yaml
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.class_utils import build_master_class_list
from src.data.merge import merge_datasets
from src.data.cleaning import (
    remove_fake_boxes,
    merge_rare_classes,
    remove_empty_classes,
    oversample_classes
)
from src.data.dataset import build_dataloaders
from src.training.train_teacher import train_teacher
from src.training.train_kd import train_kd
from src.evaluation.quantize_eval import run_full_evaluation


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / 'pipeline.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="argo_ojs full pipeline")
    parser.add_argument("--config",           default="configs/config.yaml")
    parser.add_argument("--teacher-weights",  default=None, help="Skip teacher training")
    parser.add_argument("--student-weights",  default=None, help="Skip KD training")
    parser.add_argument("--skip-merge",       action="store_true")
    parser.add_argument("--skip-teacher",     action="store_true")
    parser.add_argument("--skip-kd",          action="store_true")
    parser.add_argument("--skip-eval",        action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg    = cfg['data']
    kd_cfg      = cfg['kd']
    teacher_cfg = cfg['teacher']
    student_cfg = cfg['student']
    opt_cfg     = cfg['optimizer']
    clean_cfg   = cfg['cleaning']

    work_dir  = Path(data_cfg['work_dir'])
    merged_dir = work_dir / 'merged'
    output_dir = work_dir / 'outputs'

    setup_logging(output_dir)
    log = logging.getLogger("pipeline")

    # ----------------------------------------------------------------
    # Stage 1+2: Class Discovery + Dataset Merge
    # ----------------------------------------------------------------
    yaml_path = merged_dir / 'data.yaml'

    if not args.skip_merge:
        log.info("Stage 1: Building master class list...")
        master, cotton_remap, mhweed_remap = build_master_class_list(
            cotton_path=data_cfg['cotton_path'],
            mhweed_path=data_cfg['mhweed_path'],
            expected_nc=28
        )
        log.info(f"  Master classes ({len(master)}): {master}")

        log.info("Stage 2: Merging datasets...")
        yaml_path = merge_datasets(
            cotton_path=data_cfg['cotton_path'],
            mhweed_path=data_cfg['mhweed_path'],
            merged_dir=str(merged_dir),
            master_classes=master,
            cotton_remap=cotton_remap,
            mhweed_remap=mhweed_remap,
            seed=cfg['seed']
        )

        # ----------------------------------------------------------------
        # Stage 3: Data Cleaning
        # ----------------------------------------------------------------
        log.info("Stage 3a: Removing fake boxes...")
        remove_fake_boxes(
            merged_dir=str(merged_dir),
            area_threshold=clean_cfg['fake_box_area_threshold'],
            wh_threshold=clean_cfg['fake_box_wh_threshold']
        )

        log.info("Stage 3b: Merging rare classes...")
        merge_rare_classes(
            merged_dir=str(merged_dir),
            yaml_path=str(yaml_path),
            rare_threshold=cfg['classes']['rare_threshold']
        )

        log.info("Stage 3c: Removing empty classes...")
        remove_empty_classes(
            merged_dir=str(merged_dir),
            yaml_path=str(yaml_path)
        )

        # ----------------------------------------------------------------
        # Stage 4: Oversampling
        # ----------------------------------------------------------------
        log.info("Stage 4: Oversampling underrepresented classes...")
        oversample_classes(
            merged_dir=str(merged_dir),
            yaml_path=str(yaml_path),
            target_count=clean_cfg['oversample_target'],
            seed=cfg['seed']
        )

    with open(yaml_path) as f:
        data_yaml = yaml.safe_load(f)
    nc = data_yaml['nc']
    names = data_yaml['names']
    master_classes = list(names.values()) if isinstance(names, dict) else names
    log.info(f"Final dataset: {nc} classes")

    # ----------------------------------------------------------------
    # Stage 5: Train Teacher
    # ----------------------------------------------------------------
    teacher_weights = args.teacher_weights
    if not args.skip_teacher and teacher_weights is None:
        log.info("Stage 5: Training teacher (YOLOv8l)...")
        teacher_weights = str(train_teacher(
            yaml_path=str(yaml_path),
            output_dir=str(output_dir),
            epochs=teacher_cfg['epochs'],
            patience=teacher_cfg['patience'],
            batch=teacher_cfg['batch'],
            img_size=data_cfg['img_size'],
            device=0,
            seed=cfg['seed']
        ))

    assert teacher_weights and Path(teacher_weights).exists(), \
        "Teacher weights not found. Run without --skip-teacher or provide --teacher-weights."

    # ----------------------------------------------------------------
    # Stage 6: KD Training
    # ----------------------------------------------------------------
    student_weights_path = args.student_weights
    if not args.skip_kd and student_weights_path is None:
        log.info("Stage 6: Knowledge Distillation training (YOLOv8s)...")

        train_loader, val_loader = build_dataloaders(
            merged_dir=str(merged_dir),
            img_size=data_cfg['img_size'],
            batch_size=data_cfg['batch_size'],
            num_workers=data_cfg['num_workers'],
            nc=nc,
            seed=cfg['seed']
        )

        student_weights_path = str(train_kd(
            yaml_path=str(yaml_path),
            teacher_weights=teacher_weights,
            student_weights=student_cfg['weights'],
            master_classes=master_classes,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(output_dir),
            nc=nc,
            img_size=data_cfg['img_size'],
            batch_size=data_cfg['batch_size'],
            epochs=student_cfg['epochs'],
            patience=student_cfg['patience'],
            lr=opt_cfg['lr'],
            weight_decay=opt_cfg['weight_decay'],
            temperature=kd_cfg['temperature'],
            alpha=kd_cfg['alpha'],
            beta=kd_cfg['beta'],
            gamma=kd_cfg['gamma'],
            save_every=student_cfg['save_every'],
            check_every=student_cfg['check_every'],
            seed=cfg['seed']
        ))

    # ----------------------------------------------------------------
    # Stage 7: Quantization + Evaluation
    # ----------------------------------------------------------------
    if not args.skip_eval and student_weights_path:
        log.info("Stage 7: Quantization + full evaluation...")
        names_dict = names if isinstance(names, dict) else {i: n for i, n in enumerate(names)}
        report = run_full_evaluation(
            kd_ckpt_path=student_weights_path,
            yaml_path=str(yaml_path),
            nc=nc,
            class_names=names_dict,
            output_dir=str(output_dir / 'evaluation'),
            img_size=data_cfg['img_size'],
            batch=data_cfg['batch_size']
        )
        log.info(f"Evaluation complete. mAP50={report['fp32']['metrics']['map50']:.4f}")

    log.info("🏁 Pipeline complete.")


if __name__ == "__main__":
    main()
