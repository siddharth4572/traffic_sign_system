"""
modules/detector.py
YOLOv8-based traffic sign detector — training + inference.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import yaml
from typing import List, Dict


class TrafficSignDetector:
    """
    Wraps YOLOv8 for:
      - Training on a custom annotated traffic sign dataset.
      - Real-time inference on video frames.
    """

    def __init__(self, weights_path: str = None, cfg: dict = None):
        self.cfg = cfg or {}
        self.conf_threshold = self.cfg.get("yolo", {}).get("conf_threshold", 0.45)
        self.iou_threshold  = self.cfg.get("yolo", {}).get("iou_threshold", 0.50)
        self.img_size       = self.cfg.get("yolo", {}).get("img_size", 640)
        self.device         = self.cfg.get("yolo", {}).get("device", "cpu")

        if weights_path and Path(weights_path).exists():
            print(f"[Detector] Loading weights: {weights_path}")
            self.model = YOLO(weights_path)
        else:
            base = self.cfg.get("yolo", {}).get("base_model", "yolov8n.pt")
            print(f"[Detector] Loading base model: {base}")
            self.model = YOLO(base)

    # ── Training ──────────────────────────────────────────────────────────────
    def train(self, data_yaml: str, output_dir: str = "checkpoints/"):
        """
        Train YOLOv8 on your dataset.
        data_yaml: path to data.yaml (YOLO format with train/val/test paths + nc)
        """
        results = self.model.train(
            data=data_yaml,
            epochs=self.cfg.get("yolo", {}).get("epochs", 50),
            imgsz=self.img_size,
            batch=self.cfg.get("yolo", {}).get("batch_size", 16),
            device=self.device,
            project=output_dir,
            name="yolo_tsr",
            exist_ok=True,
            patience=10,
            save=True,
            val=True,
        )
        best_pt = Path(output_dir) / "yolo_tsr" / "weights" / "best.pt"
        print(f"[Detector] Training complete. Best weights: {best_pt}")
        return results

    # ── Inference ─────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run inference on a single BGR frame.
        Returns list of dicts: {bbox, confidence, class_id, label}
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                label = r.names.get(cls, str(cls))
                detections.append({
                    "bbox":       (x1, y1, x2, y2),
                    "confidence": conf,
                    "class_id":   cls,
                    "label":      label,
                })
        return detections

    # ── Annotation helper ─────────────────────────────────────────────────────
    @staticmethod
    def draw_detections(frame: np.ndarray, detections: list[dict],
                        hazards: list[dict] = None) -> np.ndarray:
        """Draw bounding boxes and labels on a frame."""
        vis = frame.copy()
        hazard_map = {h["class_id"]: h for h in (hazards or [])}

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cls_id = det["class_id"]
            conf   = det["confidence"]
            label  = det.get("label", str(cls_id))

            # Box color by hazard level
            hz = hazard_map.get(cls_id, {})
            level = hz.get("level", "info")
            color = {
                "critical": (60,  60,  226),
                "warning":  (39, 159,  239),
                "info":     (117, 139, 59),
            }.get(level, (128, 128, 128))   # BGR

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis

    # ── YOLO data.yaml generator ──────────────────────────────────────────────
    @staticmethod
    def generate_data_yaml(train_path: str, val_path: str,
                           nc: int, names: list, out: str = "data/yolo_data.yaml"):
        data = {
            "train": train_path,
            "val":   val_path,
            "nc":    nc,
            "names": names,
        }
        with open(out, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"[Detector] data.yaml written to {out}")
