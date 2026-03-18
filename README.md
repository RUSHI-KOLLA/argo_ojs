# argo_ojs — Weed Detection via Knowledge Distillation

> **YOLOv8l Teacher → YOLOv8s Student** | 19-class weed detection | INT8 quantization

A deep learning pipeline that compresses a high-accuracy YOLOv8l weed detector into a lightweight YOLOv8s model using Knowledge Distillation (KD), while preserving detection quality for edge deployment.

---

## Results

| Model | mAP50 | mAP50-95 | Params | Notes |
|---|---|---|---|---|
| Teacher (YOLOv8l) | 0.792 | 0.603 | ~43M | Frozen during KD |
| Student FP32 (YOLOv8s) | ~0.75+ | — | ~11M | After KD training |
| Student INT8 | — | — | ~3MB | Post-training quantization |

---

## Project Structure

```
argo_ojs/
├── notebooks/                        # Original Kaggle notebooks
│   ├── teacher-train.ipynb           # Train YOLOv8l teacher
│   ├── student-train.ipynb           # KD training loop
│   ├── kd-book.ipynb                 # Full combined pipeline
│   └── quantization-full-evaluation.ipynb
│
├── src/
│   ├── data/
│   │   ├── class_utils.py            # Class discovery & name cleaning
│   │   ├── merge.py                  # Dataset merging (Cotton + MH-Weed)
│   │   ├── cleaning.py               # Fake box removal, class merging, oversampling
│   │   └── dataset.py                # PyTorch Dataset, transforms, DataLoaders
│   │
│   ├── models/
│   │   └── architectures.py          # FrozenTeacher, Student, feature adapters
│   │
│   ├── training/
│   │   ├── kd_loss.py                # KD loss: detection + feature + KL
│   │   ├── train_teacher.py          # Teacher training script
│   │   └── train_kd.py               # KD training loop with validation
│   │
│   └── evaluation/
│       └── quantize_eval.py          # INT8 quantization + FP32/INT8 comparison
│
├── configs/
│   └── config.yaml                   # All hyperparameters & paths
│
├── scripts/
│   └── run_pipeline.py               # End-to-end pipeline runner
│
├── outputs/                          # Training outputs (gitignored)
├── requirements.txt
└── README.md
```

---

## Datasets

| Dataset | Classes | Source |
|---|---|---|
| Cotton-Weed-12 | 12 | Kaggle: `jawadulkarim117/cotton-weed-12-class` |
| MH-Weed16 | 16 | Kaggle: `sayalis069/mh-weed16` |
| **Merged** | **19** | Rare classes folded into `OtherWeed` |

### Final 19 Classes
After merging and cleaning, the unified dataset contains:
`Waterhemp`, `MorningGlory`, `Purslane`, `SpottedSpurge`, `Carpetweed`, `Ragweed`,
`Eclipta`, `PricklySida`, `PalmerAmaranth`, `Sicklepod`, `Goosegrass`, `Kena`,
`Lavhala`, `Gajar Gavat`, `Graceful Sandmart`, `Sicklepod Mh`, `Harali`,
`OtherWeed`, `Lamber Quarter Plant`

---

## Setup

```bash
git clone https://github.com/your-username/argo_ojs.git
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

### Option A — Full pipeline (from raw data to quantized model)

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

### Option B — Skip teacher training (use existing weights)

```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --teacher-weights outputs/best_teacher.pt
```

### Option C — KD only (data already merged)

```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --skip-merge \
  --teacher-weights outputs/best_teacher.pt
```

### Option D — Evaluation only

```bash
python scripts/run_pipeline.py \
  --config configs/config.yaml \
  --skip-merge --skip-teacher --skip-kd \
  --student-weights outputs/best_kd_student.pt
```

### Individual modules

```bash
# Train teacher only
python -m src.training.train_teacher --yaml merged/data.yaml --output outputs

# Run KD training only
python -m src.training.train_kd \
  --yaml merged/data.yaml \
  --teacher outputs/best_teacher.pt \
  --output outputs

# Evaluate + quantize
python -m src.evaluation.quantize_eval \
  --ckpt outputs/best_kd_student.pt \
  --yaml merged/data.yaml
```

---

## Knowledge Distillation Design

```
Input Image
    │
    ├──► Teacher (YOLOv8l, frozen)  ──► Teacher predictions
    │         │
    │    Feature hooks              ──► t_feats {l4, l6, l9}
    │
    └──► Student (YOLOv8s)          ──► Student predictions
              │
         Feature hooks              ──► s_feats {l4, l6, l9}
              │
         Adapters (1×1 conv)        ──► aligned s_feats

Total Loss = α·L_det + β·L_feat + γ·L_kl
```

| Loss Component | Weight | Description |
|---|---|---|
| `L_det` | α = 0.5 | YOLOv8 detection loss (box + cls + dfl) |
| `L_feat` | β = 0.3 | MSE between adapted student and teacher features at layers 4, 6, 9 |
| `L_kl` | γ = 0.2 | KL divergence between temperature-scaled class logits |

**Temperature:** T = 3.0 (controls softness of probability distributions)

---

## Data Cleaning Pipeline

1. **Fake box removal** — Bounding boxes covering >90% of the image are annotation errors and are removed
2. **Rare class merging** — Classes with <400 samples are merged into `OtherWeed`
3. **Empty class removal** — Classes with 0 samples after cleaning are dropped from YAML
4. **Oversampling** — Training images for underrepresented classes are duplicated to reach 750+ samples

---

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Image size | 640×640 | Standard YOLO input |
| Teacher batch | 8 | YOLOv8l memory constraint |
| Student batch | 16 | |
| KD epochs | 120 | With early stopping (patience=30) |
| Optimizer | AdamW | Student lr=1e-3, adapters lr=2e-3 |
| Scheduler | CosineAnnealing | |
| Mixed precision | ✅ | torch.amp |
| Seed | 42 | Full reproducibility |

---

## Outputs

After running the pipeline, `outputs/` will contain:

```
outputs/
├── best_teacher.pt               # Best teacher weights
├── best_kd_student.pt            # Best KD student weights
├── teacher_train/                # Teacher training run (plots, weights, metrics)
├── student_epoch*.pt             # Periodic student checkpoints
├── evaluation/
│   ├── fp32_weights.pt
│   ├── int8_weights.pt
│   └── evaluation_report.json    # Full FP32 vs INT8 comparison
└── pipeline.log
```

---

## Kaggle Note

Paths in the original notebooks are Kaggle-specific (`/kaggle/input/...`).
Update `configs/config.yaml` with your local paths before running outside Kaggle.

Teacher weights used during KD: `rahu12345/teacher/best.pt` (Kaggle dataset)

---

## Citation

If you use this project, please cite the datasets:
- Cotton-Weed-12: Kaggle dataset by `jawadulkarim117`
- MH-Weed16: Kaggle dataset by `sayalis069`
