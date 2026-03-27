"""
modules/traffic_light_detector.py
Traffic light detection using HSV color segmentation + contour analysis.
Fast and lightweight alternative to YOLO for traffic lights.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple


class TrafficLightDetector:
    """
    Detects traffic lights (red, yellow, green) using HSV color space.
    
    - Red light: H ~0-10 or 170-180, S >100, V >100
    - Yellow light: H ~15-35, S >100, V >100  
    - Green light: H ~40-80, S >100, V >100
    """

    def __init__(self, cfg: dict = None):
        self.cfg = cfg or {}
        
        # HSV ranges for each light color
        self.hsv_ranges = {
            "red": [
                {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
                {"lower": np.array([170, 100, 100]), "upper": np.array([180, 255, 255])},
            ],
            "yellow": [
                {"lower": np.array([15, 100, 100]), "upper": np.array([35, 255, 255])},
            ],
            "green": [
                {"lower": np.array([40, 100, 100]), "upper": np.array([80, 255, 255])},
            ],
        }
        
        # Morphological operations
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Min/max contour area for filtering noise
        self.min_area = 100
        self.max_area = 50000

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect traffic lights in frame.
        
        Returns:
            List of dicts with: bbox, label (red/yellow/green), confidence
        """
        detections = []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect each color
        for color, ranges in self.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for color_range in ranges:
                mask_part = cv2.inRange(hsv, color_range["lower"], color_range["upper"])
                mask = cv2.bitwise_or(mask, mask_part)
            
            # Morphological cleanup
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area or area > self.max_area:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter by aspect ratio (traffic lights are roughly square or vertical)
                aspect_ratio = float(h) / w if w > 0 else 0
                if aspect_ratio < 0.7 or aspect_ratio > 2.5:
                    continue
                
                detections.append({
                    "bbox": (x, y, x + w, y + h),
                    "label": color,
                    "confidence": 0.85,
                    "class_id": {"red": 60, "yellow": 61, "green": 62}[color],
                })
        
        # Remove duplicates (same light detected multiple times)
        detections = self._remove_duplicates(detections)
        return detections

    def _remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove overlapping detections (IoU based)."""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence descending
        detections = sorted(detections, key=lambda x: x["confidence"], reverse=True)
        
        keep = []
        for i, det in enumerate(detections):
            keep_det = True
            for kept_det in keep:
                iou = self._compute_iou(det["bbox"], kept_det["bbox"])
                if iou > iou_threshold:
                    keep_det = False
                    break
            if keep_det:
                keep.append(det)
        
        return keep

    @staticmethod
    def _compute_iou(box1: Tuple[float, float, float, float],
                     box2: Tuple[float, float, float, float]) -> float:
        """Compute intersection over union."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    @staticmethod
    def draw_detections(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw traffic light bounding boxes on frame."""
        vis = frame.copy()
        
        colors = {
            "red": (0, 0, 255),        # BGR
            "yellow": (0, 255, 255),
            "green": (0, 255, 0),
        }
        
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            label = det["label"].upper()
            color = colors.get(det["label"], (200, 200, 200))
            
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            text = f"Light: {label}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis
