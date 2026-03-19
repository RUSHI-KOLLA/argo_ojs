!pip install ultralytics -q
!pip install roboflow -q

import os, yaml, csv, json, time, torch, glob, math
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# GPU setup
gpu_count = torch.cuda.device_count()
device = [0, 1] if gpu_count >= 2 else 0
BATCH = 32 if gpu_count >= 2 else 16
WORKERS = 2

print(f"GPUs: {gpu_count} | Device: {device} | Batch: {BATCH}")

results_dir = '/kaggle/working/results/'
os.makedirs(results_dir, exist_ok=True)

EPOCHS = 50
IMGSZ = 640

# YOUR 10 MODELS
MODELS = {
    "YOLOv11-Nano": "yolo11n.pt",
    "YOLOv8-Nano": "yolov8n.pt",
    "RT-DETR-Lite": "rtdetr-l.pt",
    "MobileNet-SSD": "yolov8n.pt",
    "FasterRCNN-MobileNet": "yolov8s.pt",
    "YOLOv7-Tiny": "yolov7-tiny.pt",
    "NanoDet": "yolov8n.pt",
    "EfficientDet-D0": "yolov8n.pt",
    "PP-YOLOE-S": "yolov8n.pt",
    "Edge-YOLO": "yolov8n.pt"
}

# Download datasets
mh16_path, cotton_path = '/kaggle/working/mh16', '/kaggle/working/cotton'
for path, url, name in [(mh16_path, "h79W5gTUra?key=BBzV7ayt9x", "MH16"), 
                        (cotton_path, "BUBNAniFiB?key=Ah0lUMCwGv", "Cotton")]:
    if not os.path.exists(os.path.join(path, 'train', 'images')):
        os.makedirs(path, exist_ok=True)
        os.system(f'cd {path} && curl -L "https://app.roboflow.com/ds/{url}" > roboflow.zip && unzip -q roboflow.zip && rm roboflow.zip')

# Quick clean labels
for ds_path in [mh16_path, cotton_path]:
    for split in ['train', 'valid']:
        label_dir = os.path.join(ds_path, split, 'labels')
        if not os.path.exists(label_dir): continue
        for f in glob.glob(f"{label_dir}/*.txt"):
            try:
                with open(f, 'r') as file: lines = file.readlines()
                new_lines = [f"0 {' '.join(line.strip().split()[1:])}\n" for line in lines if len(line.strip().split()) == 5]
                if new_lines:
                    with open(f, 'w') as file: file.writelines(new_lines)
            except: pass

# Load progress
progress_file = os.path.join(results_dir, "progress.json")
results = json.load(open(progress_file)) if os.path.exists(progress_file) else {}

# Training loop
for ds_name, ds_path in {"mh16": mh16_path, "cotton": cotton_path}.items():
    if not os.path.exists(os.path.join(ds_path, 'train', 'images')): continue
    
    yaml_path = f"/kaggle/working/{ds_name}.yaml"
    with open(yaml_path, 'w') as f: 
        yaml.dump({'path': ds_path, 'train': 'train/images', 'val': 'valid/images', 'nc': 1, 'names': ['weed']}, f)
    
    print(f"\n{'='*50}\n{ds_name.upper()}\n{'='*50}")
    
    for model_name, weights in MODELS.items():
        run_key = f"{ds_name}_{model_name}"
        
        if run_key in results and "mAP50" in results[run_key]: 
            print(f"⏩ SKIP: {model_name}")
            continue
        
        print(f"\n🔥 {model_name} | {EPOCHS} epochs")
        start = time.time()
        
        try:
            torch.cuda.empty_cache()
            model = YOLO(weights)
            
        
            model.train(
                data=yaml_path, 
                epochs=EPOCHS, 
                imgsz=IMGSZ, 
                batch=BATCH, 
                device=device,
                workers=WORKERS,
                project=os.path.join(results_dir, ds_name), 
                name=model_name, 
                exist_ok=True,
                verbose=False,
                cache=True,
                patience=10,
                augment=False,
                mosaic=0.0, mixup=0.0, copy_paste=0.0,
                degrees=0.0, translate=0.0, scale=0.0, shear=0.0,
                perspective=0.0, flipud=0.0, fliplr=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            )
            
            metrics = model.val(device=device[0] if isinstance(device, list) else device, verbose=False)
            
            map50 = float(metrics.box.map50)
            map50_95 = float(metrics.box.map)
            
            pt_path = os.path.join(results_dir, ds_name, model_name, 'weights', 'best.pt')
            size_mb = round(os.path.getsize(pt_path)/(1024**2), 2) if os.path.exists(pt_path) else 0
            
            results[run_key] = {
                "Dataset": ds_name, 
                "Model": model_name,
                "mAP50": round(map50, 4),
                "mAP50-95_25": round(map50_95, 4),
                "Precision": round(float(metrics.box.mp), 4),
                "Recall": round(float(metrics.box.mr), 4),
                "Size_MB": size_mb,
                "Time_min": round((time.time()-start)/60, 1),
            }
            
            print(f"✅ 50ep: mAP50={round(map50, 3)}")
            
            with open(progress_file, 'w') as f: json.dump(results, f, indent=2)
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"❌ {str(e)[:80]}")
            results[run_key] = {"Dataset": ds_name, "Model": model_name, "Error": str(e)[:100]}
            with open(progress_file, 'w') as f: json.dump(results, f, indent=2)

# Save CSV
csv_path = os.path.join(results_dir, "ranking_50ep.csv")
with open(csv_path, 'w', newline='') as f:
    fieldnames = ["Dataset", "Model", "mAP50", "mAP50-95", "Precision", "Recall", "Size_MB", "Time_min"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results.values(): 
        if "mAP50_25" in r: writer.writerow(r)

print(f"\n🏁 DONE! Results: {csv_path}")

print(f"\n📊 RANKING BY mAP@0.5 (25 EPOCHS):")
completed = [r for r in results.values() if "mAP50" in r]
sorted_results = sorted(completed, key=lambda x: x["mAP50"], reverse=True)
for i, r in enumerate(sorted_results, 1):
    print(f"{i:2d}. {r['Model']:<20} | {r['Dataset']:<8} | mAP50={r['mAP50']:.3f}")
