# Traffic AI Dashboard

A real-time traffic management system built with YOLOv8 for vehicle detection and tracking, combined with machine learning regression models to predict traffic flow patterns.

---

## Features

- **Real-time vehicle detection** using YOLOv8 Nano on webcam or uploaded video files
- **Object tracking** with ByteTrack/BoT-SORT — assigns a unique ID to each vehicle and counts it only once as it passes through the frame, preventing duplicate counts
- **Traffic flow prediction** using 3 regression models trained on collected data: Linear Regression, Polynomial Regression (degree 3), and Random Forest — with an auto-generated comparison table (MSE / RMSE / MAE / R²) and interpretation
- **Automated training pipeline** — from video input to auto-labeling, dataset preparation, and fine-tuning, all accessible from the dashboard
- **EDA module** — dataset label distribution chart, training loss analysis with automatic Overfitting/Underfitting diagnosis, fine-tuning comparison across all runs, and Confusion Matrix with per-class analysis
- **Realtime FPS display** — overlaid on the video frame and shown in the KPI panel

---

## Project Structure

```
DLCK/
├── data/
│   ├── videos/              # Input videos for frame extraction
│   ├── images/              # Extracted frames
│   └── dataset/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       ├── labels/
│       │   ├── train/
│       │   └── val/
│       └── data.yaml        # YOLO dataset config
├── results/
│   ├── traffic_model/       # Latest training result
│   ├── traffic_data.csv     # Live detection logs
│   └── test_video_result.csv
├── runs/
│   └── detect/
│       └── results/
│           ├── traffic_model/
│           ├── traffic_model-2/
│           └── ...          # All fine-tuning checkpoints
├── src/
│   ├── 1_extract_frames.py  # Cut frames from video
│   ├── 2_train_yolo.py      # Fine-tuning pipeline
│   ├── 3_detect_track.py    # Standalone detection script
│   ├── 4_prediction_models.py
│   ├── 5_app_dashboard.py   # Main entry point
│   ├── 6_auto_label.py      # Auto-labeling with YOLOv8m
│   ├── 7_prepare_dataset.py # Dataset split and yaml generation
│   ├── utils.py             # Shared utilities and chart functions
│   └── eda.py               # EDA rendering module
├── venv/
├── yolov8n.pt               # Nano model — used for training
├── yolov8m.pt               # Medium model — used for auto-labeling
└── requirements.txt
```

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Object Detection & Tracking | Ultralytics YOLOv8 |
| Video / Image Processing | OpenCV |
| Web Dashboard | Streamlit |
| Regression Models | scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| GPU Acceleration | PyTorch + CUDA 12.1 |

---

## System Requirements

- Python 3.9+
- NVIDIA GPU with CUDA support (recommended: VRAM >= 4 GB)
- RAM >= 8 GB
- Windows 10 / 11

---

## Installation

**1. Activate virtual environment**

```bash
cd DLCK
python -m venv venv
.\venv\Scripts\activate
```

**2. Install PyTorch with CUDA support**

```bash
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**3. Install remaining dependencies**

```bash
pip install -r requirements.txt
```

**4. Verify CUDA is working**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Expected output: True / Your GPU name
```

---

## Running the Dashboard

```bash
.\venv\Scripts\activate
streamlit run src/5_app_dashboard.py
```

The browser will open automatically at `http://localhost:8501`.

---

## Pipeline Overview

The dashboard sidebar guides you through 6 steps:

**Step 1 — Import Video & Extract Frames**
Drop a `.mp4` or `.avi` file into the sidebar. The system uses OpenCV to read the video and saves one frame every 0.5 seconds into `data/images/`. Frames that already exist are skipped.

**Step 2 — Auto-Label Data**
YOLOv8 Medium (`yolov8m.pt`) scans each image and detects 6 vehicle classes using the COCO standard class IDs. Results are saved as `.txt` files in YOLO format inside `data/dataset/labels/`. Images that already have a label file are skipped.

**Step 2.5 — Prepare Dataset**
The script pairs each image with its label file, shuffles the list, splits it 80% train / 20% val, copies the files into the correct folders, and generates `data.yaml` for YOLO.

**Step 3 — Train AI**
The training script automatically finds the latest `best.pt` checkpoint (by number in the folder name) and uses it as the starting point for fine-tuning. If no checkpoint exists, it trains from `yolov8n.pt` from scratch. Key training settings: `freeze=5`, `lr0=0.001`, `warmup_epochs=3`, `patience=15`, `device=0` (GPU).

**Step 5 — Test Video & Predict**
Upload a test video. The dashboard runs `model.track()` on every frame (ByteTrack), accumulates unique vehicle counts by track ID, logs data to CSV every N seconds, and displays a live prediction chart from 3 regression models alongside FPS metrics.

**Step 6 — EDA & Model Analysis**
Four tabs: label distribution bar chart, training loss chart with automatic Overfitting diagnosis, fine-tuning comparison across all runs (mAP50 progress), and Confusion Matrix with Precision/Recall analysis.

---

## Detected Vehicle Classes

The model detects 6 classes from the COCO dataset:

```
ID 0  →  Person
ID 1  →  Bicycle
ID 2  →  Car
ID 3  →  Motorcycle
ID 5  →  Bus
ID 7  →  Truck
```

Classes 4 (airplane) and 6 (train) are included in `data.yaml` to preserve the original COCO class IDs and avoid index mismatches, but are not detected in practice.

---

## Training Results

The model was trained from scratch using YOLOv8 Nano and progressively improved through 13 rounds of fine-tuning on a dataset collected from real traffic footage.

| Run | Epochs | mAP50 | Change |
|---|---|---|---|
| Baseline (from scratch) | 50 | 0.487 | — |
| Fine-tune 1 | 17 | 0.788 | +0.301 |
| Fine-tune 7 | 50 | 0.774 | — |
| Fine-tune 10 | 30 | 0.821 | +0.047 |
| Fine-tune 11 | 21 | 0.829 | +0.008 |
| Fine-tune 12 | 30 | 0.832 | +0.003 |
| **Fine-tune 13** | **30** | **0.836** | **Best** |

**Best model: mAP50 = 0.836**, Precision = 0.876, Recall = 0.721

The jump from 0.487 to 0.836 demonstrates the effectiveness of iterative fine-tuning — each round builds on the previous checkpoint rather than retraining from scratch, significantly reducing training time while continuing to improve accuracy.

---

## Known Limitations

- Recall = 0.721 means roughly 28% of real vehicles are missed, primarily due to class imbalance (car labels dominate at ~50% of the dataset)
- Adding large amounts of new data (truck/bicycle) in later fine-tuning rounds caused partial Catastrophic Forgetting — mAP50 dropped from 0.836 to ~0.70 in subsequent runs
- The regression models (Linear, Polynomial, Random Forest) predict based on collected CSV data and require at least 3 log intervals before charts appear

---

## Notes

- `5_app_dashboard.py` requires `utils.py` and `eda.py` to be in the same `src/` directory
- The dashboard always loads the latest `best.pt` automatically by reading the highest number in the `traffic_model-X` folder name — no manual path changes needed
- When adding new training data: run Step 2 (labels only for unlabeled images) → Step 2.5 → Step 3
- Avoid chaining too many fine-tuning rounds on a drastically different dataset distribution — this risks Catastrophic Forgetting of previously learned features
