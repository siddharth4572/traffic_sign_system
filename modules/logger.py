"""
modules/logger.py
Lightweight CSV + console logger for pipeline metrics.
"""

import csv
import os
import time
from pathlib import Path
from datetime import datetime


class PipelineLogger:

    def __init__(self, log_dir: str = "logs/"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path = Path(log_dir) / f"session_{timestamp}.csv"
        self._file = open(self._csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "frame", "timestamp", "fps",
            "num_detections", "top_sign", "top_conf",
            "hazard_level", "hazard_message",
        ])
        print(f"[Logger] Logging to {self._csv_path}")

    def log_frame(self, frame_id: int, detections: list,
                  alerts: list, fps: float):
        top_sign = top_conf = hazard_level = hazard_msg = ""
        if detections:
            best = max(detections, key=lambda d: d.get("confidence", 0))
            top_sign = best.get("label", "")
            top_conf = f"{best.get('confidence', 0):.3f}"
        if alerts:
            hazard_level = alerts[0]["level"]
            hazard_msg   = alerts[0]["message"]

        self._writer.writerow([
            frame_id, time.time(), f"{fps:.2f}",
            len(detections), top_sign, top_conf,
            hazard_level, hazard_msg,
        ])

    def close(self):
        self._file.flush()
        self._file.close()
        print(f"[Logger] Session saved to {self._csv_path}")
