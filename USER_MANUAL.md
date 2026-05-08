# User Manual

## 1. Overview

The Traffic Sign Recognition & Hazard Prediction System detects traffic signs, classifies them, and generates hazard alerts in real time using YOLOv8 + CNN.

## 2. Prerequisites

- Python 3.9+
- pip
- Webcam (for live mode) or a video file (for file mode)

## 3. Installation

From the project root:

```bash
pip install -r requirements.txt
```

## 4. Dataset Setup (GTSRB)

You can download the dataset manually:

```bash
python -c "from torchvision.datasets import GTSRB; GTSRB('data/raw', split='train', download=True)"
```

## 5. Training

Run from project root:

### CNN training
```bash
python train.py --mode cnn
```

### YOLO training
```bash
python train.py --mode yolo
```

### Full training
```bash
python train.py --mode all
```

## 6. Running Inference

### A) Live camera in OpenCV window
```bash
python run.py
```
Press `q` to exit.

### B) Dashboard mode
```bash
python run.py --mode dashboard
```
Then open: `http://localhost:5000`

### C) Video file input
```bash
python run.py --source /path/to/video.mp4 --mode dashboard
```

## 7. Configuration

Main configuration file:

`configs/config.yaml`

Common settings:
- `yolo.device`, `cnn.device`: set to `cuda` for NVIDIA GPU acceleration
- `inference.frame_skip`: increase for faster processing
- `inference.resize_factor`: reduce to improve speed
- `hazard.tts_enabled`: enable/disable text-to-speech alerts

## 8. Outputs

- Checkpoints are saved to the path configured in `configs/config.yaml` (`paths.checkpoints`).
- Dashboard displays live detections, alerts, and metrics.
- Console output shows pipeline startup and runtime status.

## 9. Troubleshooting

- **Module import errors**: Ensure dependencies are installed with `pip install -r requirements.txt`.
- **No camera feed**: Check webcam permissions and update `inference.camera_index` in `configs/config.yaml`.
- **Dashboard not loading**: Confirm `run.py --mode dashboard` is running, then visit `http://localhost:5000`.
- **Slow inference**: Increase `inference.frame_skip` and reduce `inference.resize_factor`.
- **GPU not used**: Verify CUDA availability and set `yolo.device`/`cnn.device` to `cuda`.

## 10. Additional Documentation

- `README.md` for project architecture and technical details
- `QUICK_START.md` for a compact setup path
