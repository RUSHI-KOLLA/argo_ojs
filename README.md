# argo_ojs — Weed Detection via Knowledge Distillation

> **YOLOv8l Teacher → YOLOv8s Student** | 19-class weed detection | FP16 + INT8 multi-format compression

A deep learning pipeline that compresses a high-accuracy YOLOv8l weed detector into a lightweight YOLOv8s model using Knowledge Distillation (KD), then exports it in multiple quantized formats for edge deployment.

---

## Results

| Model | mAP50 | FPS | Notes |
| --- | --- | --- | --- |
| Teacher (YOLOv8l) | 0.7466 | ~22 FPS | Frozen during KD; FP32 |
| Student + KD | **0.7881** | ~96 FPS | After KD training; FP32 |
| Student + KD (TRT-FP16) | **0.7893** | ~231 FPS | TensorRT FP16 post-training quantization |

<img width="2684" height="755" alt="model_comparison" src="https://github.com/user-attachments/assets/2bd588e3-f5b5-46b6-b38c-86e6d4cca3f0" />

---

## Project Structure

```
argo_ojs/
├── notebooks/
│   ├── teacher-train.ipynb                 # Train YOLOv8l teacher (150 epochs)
│   ├── student-train.ipynb                 # KD training: detection + feature loss
│   ├── kd-book.ipynb                       # Full combined pipeline
│   └── quantization-full-evaluation.ipynb  # FP16/INT8/TRT export + full evaluation
│
├── src/
│   ├── data/
│   │   ├── class_utils.py      # Class discovery & name cleaning
│   │   ├── merge.py            # Dataset merging (Cotton-Weed + MH-Weed16)
│   │   ├── cleaning.py         # Fake box removal, rare class merging, oversampling
│   │   └── dataset.py          # PyTorch Dataset, YOLO-scale transforms, DataLoaders
│   │
│   ├── models/
│   │   └── architectures.py    # FrozenTeacher, Student (YOLOv8s), FeatureAdapters
│   │
│   ├── training/
│   │   ├── kd_loss.py          # Two-component KD loss: detection + feature MSE
│   │   ├── train_teacher.py    # Teacher training script
│   │   └── train_kd.py         # KD training loop with validation
│   │
│   └── evaluation/
│       └── quantize_eval.py    # Multi-format export + benchmarking
│
├── configs/config.yaml         # All hyperparameters and paths
├── scripts/run_pipeline.py     # End-to-end pipeline runner
├── outputs/                    # Training outputs (gitignored)
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Classes | Cameras / Source |
|---|---|---|
| Cotton-Weed-12 | 12 | Standard RGB — Kaggle: `jawadulkarim117/cotton-weed-12-class` |
| MH-Weed16 | 16 | Canon, iPhone, Intel RealSense — Kaggle: `sayalis069/mh-weed16` |
| **Merged** | **19** | Rare classes (<400 samples) folded into `OtherWeed` |

### Final 19 Classes

`Waterhemp`, `MorningGlory`, `Purslane`, `SpottedSpurge`, `Carpetweed`, `Ragweed`,
`Eclipta`, `PricklySida`, `PalmerAmaranth`, `Sicklepod`, `Goosegrass`, `Kena`,
`Lavhala`, `Gajar Gavat`, `Graceful Sandmart`, `Sicklepod Mh`, `Harali`,
`OtherWeed`, `Lamber Quarter Plant`

---

## Setup

```bash
git clone https://github.com/RUSHI-KOLLA/argo_ojs.git
cd argo_ojs
pip install -r requirements.txt
```

Update dataset paths in `configs/config.yaml`:

```yaml
data:
  cotton_path: /path/to/cotton-weed-12-class
  mhweed_path: /path/to/mh-weed16/MH-Weed16
  work_dir:    /path/to/working/directory
```

---

## Usage

### Full pipeline (raw data → quantized model)
```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

### Skip teacher training (use existing weights)
```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --teacher-weights outputs/best_teacher.pt
```

### KD only (data already merged)
```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --skip-merge \
  --teacher-weights outputs/best_teacher.pt
```

### Evaluation only
```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --skip-merge --skip-teacher --skip-kd \
  --student-weights outputs/best_kd_student.pt
```

---

## Knowledge Distillation Design

```
Input Image (640×640, pixels in [0, 1])
    │
    ├──► Teacher (YOLOv8l, frozen, ~43M params)
    │         │
    │    Hooks at backbone layers 4, 6, 9
    │         └──► t_feats = {l4, l6, l9}
    │
    └──► Student (YOLOv8s, ~11M params)
              │
         Hooks at backbone layers 4, 6, 9
              └──► s_feats = {l4, l6, l9}
                        │
                   1×1 Conv Adapters (channel alignment)
                        │
                   aligned_s_feats

Total Loss = α · L_det  +  β · L_feat
           = 1.0 · L_det  +  0.05 · L_feat
```

### Loss Components

| Component | Weight | Description |
|---|---|---|
| `L_det` | α = 1.0 | Standard YOLOv8 detection loss (box=7.5, cls=0.5, dfl=1.5) |
| `L_feat` | β = 0.05 | MSE between adapter-aligned student and frozen teacher features at backbone layers 4, 6, 9 |

> **Note:** KL soft-label loss (gamma) was evaluated but disabled (gamma = 0.0) — the `kl_loss` function returns 0.0 as a fixed value. Only detection loss + feature distillation are active in training.

### Feature Adapters

Learnable 1×1 Conv layers (`nn.Conv2d`) bridge the channel dimension mismatch between YOLOv8l (teacher) and YOLOv8s (student) at each hook layer. Initialized with Xavier uniform weights and zero bias.

---

## Data Pipeline

### 1. Dataset Fusion
Cotton-Weed-12 (YOLO format, pre-split) and MH-Weed16 (Canon/iPhone/RealSense images with YOLO_darknet labels) are merged into a unified 19-class dataset. Class names are auto-discovered from YAML configs and folder structures. The duplicate `Sicklepod` class is resolved by renaming MH-Weed's version to `Sicklepod Mh`. All bounding box labels are remapped to unified class IDs.

MH-Weed16 is split 80/10/10 (train/val/test). Cotton-Weed uses its original splits.

### 2. Data Cleaning
- **Fake box removal:** Boxes with `w×h > 0.90` OR both `w > 0.95` and `h > 0.95` are removed as annotation errors
- **Rare class merging:** Two rounds — classes below 400 samples merged into `OtherWeed` (classes 18,19,23,25,26,27), then classes 11,21,22 also merged
- **Empty class removal:** `Little Mallow` (class 19 post-merge) had 0 samples and was removed, giving the final 19 classes
- **Oversampling:** Minority classes duplicated to reach 500+ training samples using random copy with unique stems (`os_{class}_{idx}`)

### 3. Augmentation (YOLO-specific Fix)

Standard KD pipelines apply ImageNet normalization which silently breaks YOLO models. This pipeline uses YOLO-compatible scaling:

```python
# Correct: scale only to [0, 1], no mean/std shift
A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0)
```

Training augmentations: HorizontalFlip (p=0.5), RandomBrightnessContrast (p=0.2), HueSaturationValue (p=0.2)

---

## Quantization & Export

The evaluation notebook exports the KD student in 4 formats and runs full benchmarking:

| Format | Method | Target |
|---|---|---|
| FP32 PyTorch | Baseline | GPU/CPU |
| **FP16 TorchScript** | `export(format='torchscript', half=True)` | GPU (primary export) |
| ONNX FP32 | `export(format='onnx', simplify=True)` | Any runtime |
| TensorRT FP16 | `export(format='engine', half=True)` | NVIDIA GPU |
| PyTorch Dynamic INT8 | `torch.quantization.quantize_dynamic(...)` | CPU |

Evaluation outputs per model: mAP50, mAP50-95, Precision, Recall, per-class AP50, model size (MB), latency (ms), FPS. Results saved to `paper_results.json`.

---

## Key Hyperparameters

| Parameter | Teacher | Student (KD) |
|---|---|---|
| Architecture | YOLOv8l | YOLOv8s |
| Image size | 640×640 | 640×640 |
| Batch size | 8 | 16 |
| Epochs | 150 | 80 |
| Patience (early stop) | 40 | 30 |
| Optimizer | AdamW | AdamW |
| LR | 0.01 | 1e-4 |
| Adapter LR | — | 2e-4 (2× student) |
| Scheduler | CosineAnnealing | CosineAnnealing |
| KD α (detection) | — | 1.0 |
| KD β (feature MSE) | — | 0.05 |
| KD γ (KL) | — | 0.0 (disabled) |
| Mixed precision (AMP) | ✅ | ✅ |
| Seed | 42 | 42 |

---

## Outputs

```
outputs/
├── best_teacher.pt                    # Best teacher weights (YOLOv8l)
├── best_kd_student.pt                 # Best KD student weights (YOLOv8s)
├── teacher_clean/weights/best.pt      # Teacher training run
├── student_fp32_deployable.pt         # Deployable FP32 YOLO model
├── student_int8_dynamic.pt            # INT8 quantized weights
├── qualitative_results.png            # Student predictions on 8 test images
├── teacher_vs_student.png             # Teacher vs Student side-by-side
├── model_comparison.png               # mAP / size / FPS bar charts
├── paper_results.json                 # Full metrics for paper tables
└── pipeline.log
```


## Authors

**Rushi Kolla**
- GitHub: [@RUSHI-KOLLA](https://github.com/RUSHI-KOLLA)
- Email: kollarushi2006@gmail.com

---

## Citation

If you use this project, please cite the datasets:
- Cotton-Weed-12: Kaggle dataset by `jawadulkarim117`
- MH-Weed16: Kaggle dataset by `sayalis069`
