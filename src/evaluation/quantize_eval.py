"""
src/evaluation/quantize_eval.py
--------------------------------
Post-training quantization and full model evaluation pipeline.

Steps:
  1. Load KD student checkpoint and build deployable YOLO model
  2. Run FP32 baseline evaluation
  3. Apply INT8 static quantization (CPU) or torch dynamic quantization
  4. Compare FP32 vs INT8: mAP, latency, model size
  5. Save quantized model + evaluation report (JSON)
"""

import copy
import json
import time
import shutil
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


# ----------------------------------------------------------------
# Build Deployable YOLO from KD Checkpoint
# ----------------------------------------------------------------

def build_yolo_from_kd_ckpt(
    ckpt_path: str,
    nc: int,
    class_names: dict,
    device: str = 'cuda'
) -> YOLO:
    """
    Load a KD student checkpoint and wrap it in a YOLO object
    for native ultralytics inference and validation.

    Args:
        ckpt_path: Path to saved KD checkpoint (.pt).
        nc: Number of classes.
        class_names: Dict mapping class_id → class_name.
        device: 'cuda' or 'cpu'.

    Returns:
        Configured YOLO model object.
    """
    print("  Loading KD checkpoint...")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model', ckpt)

    print("  Building YOLOv8s architecture...")
    model = DetectionModel(cfg="yolov8s.yaml", ch=3, nc=nc)

    # Adapt detection head
    det = model.model[-1]
    det.nc = nc
    for b in det.cv3:
        old = b[-1]
        new_conv = nn.Conv2d(
            old.in_channels, nc,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=True
        )
        nn.init.normal_(new_conv.weight, 0, 0.01)
        nn.init.constant_(new_conv.bias, -4.5)
        b[-1] = new_conv
    det.no = det.nc + det.reg_max * 4

    model.load_state_dict(state_dict, strict=False)
    model.nc = nc
    model.names = class_names
    model.eval()

    y = YOLO('yolov8s.pt')
    y.model = model.to(device)
    return y


# ----------------------------------------------------------------
# FP32 Evaluation
# ----------------------------------------------------------------

def evaluate_model(
    yolo_model: YOLO,
    yaml_path: str,
    img_size: int = 640,
    batch: int = 16,
    device: str = 'cuda'
) -> dict:
    """
    Run YOLO native validation and return metrics dict.

    Returns:
        Dict with keys: map50, map50_95, precision, recall
    """
    results = yolo_model.val(
        data=str(yaml_path),
        imgsz=img_size,
        batch=batch,
        device=0 if 'cuda' in device else 'cpu',
        verbose=False,
        plots=False
    )

    return {
        'map50':     float(results.box.map50),
        'map50_95':  float(results.box.map),
        'precision': float(results.box.mp),
        'recall':    float(results.box.mr),
    }


# ----------------------------------------------------------------
# Latency Benchmark
# ----------------------------------------------------------------

def benchmark_latency(
    model: nn.Module,
    img_size: int = 640,
    n_warmup: int = 20,
    n_runs: int = 100,
    device: str = 'cuda'
) -> dict:
    """
    Measure average inference latency and throughput.

    Args:
        model: PyTorch model (eval mode).
        img_size: Input image size.
        n_warmup: Warmup iterations (excluded from timing).
        n_runs: Timed iterations.
        device: 'cuda' or 'cpu'.

    Returns:
        Dict with avg_ms, std_ms, fps.
    """
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    times = []

    with torch.no_grad():
        # Warmup
        for _ in range(n_warmup):
            _ = model(dummy)

        # Timed runs
        for _ in range(n_runs):
            if 'cuda' in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy)
            if 'cuda' in device:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return {
        'avg_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'fps':    float(1000.0 / np.mean(times))
    }


# ----------------------------------------------------------------
# Dynamic Quantization
# ----------------------------------------------------------------

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply PyTorch dynamic INT8 quantization to Linear and Conv2d layers.
    Works on CPU only.

    Args:
        model: FP32 PyTorch model.

    Returns:
        Quantized model (CPU).
    """
    model_cpu = copy.deepcopy(model).cpu()
    quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized


# ----------------------------------------------------------------
# Model Size
# ----------------------------------------------------------------

def get_model_size_mb(model: nn.Module, path: Path) -> float:
    """Save model to disk and return size in MB."""
    torch.save(model.state_dict(), path)
    size_mb = path.stat().st_size / (1024 ** 2)
    return round(size_mb, 2)


# ----------------------------------------------------------------
# Full Evaluation Pipeline
# ----------------------------------------------------------------

def run_full_evaluation(
    kd_ckpt_path: str,
    yaml_path: str,
    nc: int,
    class_names: dict,
    output_dir: str,
    img_size: int = 640,
    batch: int = 16
) -> dict:
    """
    Run the complete evaluation pipeline:
      1. FP32 accuracy evaluation
      2. FP32 latency benchmark
      3. INT8 dynamic quantization
      4. INT8 latency benchmark
      5. Model size comparison
      6. Save JSON report

    Args:
        kd_ckpt_path: Path to KD student checkpoint.
        yaml_path: Path to YOLO data.yaml.
        nc: Number of classes.
        class_names: Dict of class id → name.
        output_dir: Directory for report and saved models.
        img_size: Input image size.
        batch: Batch size for validation.

    Returns:
        Full evaluation report dict.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("📊 FULL EVALUATION: FP32 vs INT8 Quantized")
    print("=" * 70)

    # --- FP32 ---
    print("\n[1/5] Building FP32 model...")
    fp32_yolo = build_yolo_from_kd_ckpt(kd_ckpt_path, nc, class_names, device)
    fp32_model = fp32_yolo.model

    print("[2/5] FP32 evaluation...")
    fp32_metrics = evaluate_model(fp32_yolo, yaml_path, img_size, batch, device)
    print(f"  mAP50={fp32_metrics['map50']:.4f}  mAP50-95={fp32_metrics['map50_95']:.4f}")

    print("[3/5] FP32 latency benchmark...")
    fp32_latency = benchmark_latency(fp32_model, img_size, device=device)
    print(f"  Avg: {fp32_latency['avg_ms']:.2f}ms ± {fp32_latency['std_ms']:.2f}ms  ({fp32_latency['fps']:.1f} FPS)")

    fp32_size = get_model_size_mb(fp32_model, output / 'fp32_weights.pt')

    # --- INT8 ---
    print("[4/5] Applying dynamic quantization...")
    int8_model = apply_dynamic_quantization(fp32_model)
    int8_size = get_model_size_mb(int8_model, output / 'int8_weights.pt')

    print("[5/5] INT8 latency benchmark (CPU)...")
    int8_latency = benchmark_latency(int8_model, img_size, device='cpu')
    print(f"  Avg: {int8_latency['avg_ms']:.2f}ms ± {int8_latency['std_ms']:.2f}ms  ({int8_latency['fps']:.1f} FPS)")

    # --- Report ---
    report = {
        'fp32': {
            'metrics':  fp32_metrics,
            'latency':  fp32_latency,
            'size_mb':  fp32_size,
        },
        'int8': {
            'latency':  int8_latency,
            'size_mb':  int8_size,
        },
        'compression': {
            'size_reduction_pct': round((1 - int8_size / fp32_size) * 100, 1),
            'speedup_cpu':        round(fp32_latency['avg_ms'] / max(int8_latency['avg_ms'], 0.001), 2),
        }
    }

    report_path = output / 'evaluation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report saved: {report_path}")
    print(f"   Size reduction: {report['compression']['size_reduction_pct']}%")
    print(f"   CPU speedup:    {report['compression']['speedup_cpu']}x")

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantization + Evaluation")
    parser.add_argument("--ckpt",   required=True, help="KD student checkpoint")
    parser.add_argument("--yaml",   required=True, help="data.yaml path")
    parser.add_argument("--nc",     type=int, default=19)
    parser.add_argument("--output", default="outputs/eval")
    args = parser.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg['names']
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}

    run_full_evaluation(
        kd_ckpt_path=args.ckpt,
        yaml_path=args.yaml,
        nc=args.nc,
        class_names=names,
        output_dir=args.output
    )
