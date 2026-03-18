# Traffic Sign Recognition & Hazard Prediction System

Real-time traffic sign detection, classification, and driver hazard alerting using YOLOv8 + CNN.

---

## Project Structure

```
traffic_sign_system/
├── configs/
│   ├── config.yaml          # All hyperparameters & paths
│   └── sign_classes.py      # 43-class labels + hazard rules
├── models/
│   └── cnn_classifier.py    # ResNet50/EfficientNet/ViT + trainer
├── modules/
│   ├── detector.py          # YOLOv8 detection wrapper
│   ├── preprocessor.py      # GTSRB dataset + augmentation pipeline
│   ├── hazard_engine.py     # Rule-based hazard prediction + TTS alerts
│   ├── pipeline.py          # Real-time inference orchestrator
│   └── logger.py            # CSV session logger
├── ui/
│   └── dashboard.py         # Flask + SocketIO live dashboard
├── utils/
│   └── evaluate.py          # Metrics, confusion matrix, reports
├── train.py                 # Training entrypoint
├── run.py                   # Inference entrypoint
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the GTSRB dataset
```bash
# Option A: torchvision (auto-download)
python -c "from torchvision.datasets import GTSRB; GTSRB('data/raw', split='train', download=True)"

# Option B: Kaggle CLI
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign
unzip gtsrb-german-traffic-sign.zip -d data/raw/
```

### 3. Train

```bash
# Train CNN classifier only (recommended first step)
python train.py --mode cnn

# Train YOLO detector (requires annotated YOLO-format dataset)
python train.py --mode yolo

# Train both
python train.py --mode all
```

### 4. Run real-time inference

```bash
# OpenCV window (live camera)
python run.py

# Web dashboard (open http://localhost:5000)
python run.py --mode dashboard

# Run on a video file
python run.py --source path/to/video.mp4 --mode dashboard
```

---

## System Architecture

```
Camera Frame
    │
    ▼
Preprocessing (resize, normalize, augment)
    │
    ├──▶ YOLOv8 Detection  ──▶  Bounding Boxes + Confidence
    │                                      │
    │                                      ▼
    └──────────────────────▶  CNN Classifier (43 classes)
                                           │
                                           ▼
                               Hazard Prediction Engine
                               (rule-based, temporal smoothing)
                                           │
                          ┌────────────────┼────────────────┐
                          ▼                ▼                ▼
                    Dashboard UI      Alert / TTS       CSV Logger
```

---

## Model Details

### YOLOv8 Detector
- Base: `yolov8n.pt` (nano, ~3.2M params) — swap to `yolov8s` or `yolov8m` for higher accuracy
- Input: 640×640
- Output: bounding boxes with confidence scores
- Training: fine-tuned on GTSRB with YOLO annotations

### CNN Classifier
- Backbone: ResNet50 (configurable to EfficientNet-B0 or ViT-B/16)
- Input: 224×224 RGB
- Output: 43-class softmax
- Augmentation: brightness/contrast jitter, Gaussian noise, motion blur, perspective

### Hazard Engine
- Pure rule-based: maps sign class → hazard level (critical / warning / info)
- Temporal smoothing: majority vote over a rolling 3-frame window
- Alert cooldown: suppresses repeated alerts for the same sign (configurable)
- TTS output via `pyttsx3` (optional)

---

## Hazard Levels

| Level    | Examples                              | Action                      |
|----------|---------------------------------------|-----------------------------|
| CRITICAL | Stop sign, No entry                   | Full stop / do not proceed  |
| WARNING  | Yield, Slippery road, Children ahead  | Slow down, stay alert       |
| INFO     | Speed limits, Mandatory directions    | Comply with sign            |

---

## Performance Targets

| Metric              | Target         |
|---------------------|----------------|
| Detection mAP@0.5   | ≥ 85%          |
| Classification Top-1| ≥ 95%          |
| Real-time FPS       | ≥ 20 FPS       |
| Alert latency       | < 200 ms       |

---

## Configuration

All settings in `configs/config.yaml`. Key options:

```yaml
yolo:
  conf_threshold: 0.45   # detection confidence cutoff
  epochs: 50

cnn:
  backbone: "resnet50"   # or efficientnet_b0, vit_b_16
  epochs: 30
  learning_rate: 0.001

hazard:
  alert_cooldown_sec: 3
  tts_enabled: true

inference:
  target_fps: 30
  camera_index: 0        # or path to video file
```

---

## References

1. Wang et al. — Transformer-based vision models for traffic sign recognition
2. Zeng et al. — Real-time detection via YOLOv5 + lightweight attention mechanisms
3. GTSRB Dataset — Stallkamp et al., IJCNN 2011
