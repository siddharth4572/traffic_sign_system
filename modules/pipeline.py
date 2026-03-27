"""
modules/pipeline.py
Real-time inference pipeline: Camera → YOLO → CNN → Hazard Engine → Annotated Frame
Speed optimized with frame skipping, frame resizing, and lightweight processing.
"""

import cv2
import time
import threading
import queue
import numpy as np
from pathlib import Path

from modules.detector       import TrafficSignDetector
from models.cnn_classifier  import TrafficSignClassifier
from modules.hazard_engine  import HazardEngine
from modules.preprocessor   import FramePreprocessor
from modules.logger         import PipelineLogger
from configs.sign_classes   import SIGN_LABELS


class RealtimePipeline:
    """
    Orchestrates the real-time pipeline with speed optimizations.

    Usage:
        pipeline = RealtimePipeline(cfg)
        pipeline.run()          # blocking, shows OpenCV window
        # or
        for frame_data in pipeline.iter_frames():
            ...                 # non-blocking generator for Flask streaming
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        inf = cfg.get("inference", {})
        self.target_fps   = inf.get("target_fps", 30)
        self.cam_index    = inf.get("camera_index", 0)
        self.buf_size     = inf.get("frame_buffer_size", 3)  # Reduced for speed
        
        # ── Performance optimization settings ──────────────────────────────────
        self.frame_skip   = inf.get("frame_skip", 1)  # Process every Nth frame
        self.resize_factor = inf.get("resize_factor", 1.0)  # 0.5 = 50% smaller
        self.skip_cnn_classification = inf.get("skip_cnn_classification", False)

        # ── Load models ───────────────────────────────────────────────────────
        yolo_w = cfg["paths"]["yolo_weights"]
        cnn_w  = cfg["paths"]["cnn_weights"]

        self.detector    = TrafficSignDetector(yolo_w, cfg)
        self.classifier  = TrafficSignClassifier(cnn_w, cfg)
        self.preprocessor = FramePreprocessor(cfg["cnn"]["img_size"])
        self.hazard      = HazardEngine(cfg)
        self.logger      = PipelineLogger(cfg["paths"]["logs"])

        # ── State ─────────────────────────────────────────────────────────────
        self.running = False
        self._frame_queue: queue.Queue = queue.Queue(maxsize=self.buf_size)
        self._latest_annotated: np.ndarray = None
        self._frame_count_total = 0
        self._stats = {
            "fps": 0.0, 
            "detections": 0, 
            "frame_count": 0, 
            "current_dets": [],
        }

    # ── Camera reader thread ──────────────────────────────────────────────────
    def _camera_reader(self, cap: cv2.VideoCapture):
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if not self._frame_queue.full():
                self._frame_queue.put(frame)

    # ── Process one frame ─────────────────────────────────────────────────────
    def _process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list, list]:
        # Resize frame if needed (for speed)
        if self.resize_factor < 1.0:
            h_scaled = int(frame.shape[0] * self.resize_factor)
            w_scaled = int(frame.shape[1] * self.resize_factor)
            frame_scaled = cv2.resize(frame, (w_scaled, h_scaled), interpolation=cv2.INTER_LINEAR)
        else:
            frame_scaled = frame
        
        # 1. Detect signs (fastest path first)
        detections = self.detector.detect(frame_scaled)

        # 2. Classify each sign crop (optional - skip for speed)
        classified = []
        if not self.skip_cnn_classification:
            for det in detections:
                tensor = self.preprocessor.preprocess_crop(frame_scaled, det["bbox"])
                if tensor is None:
                    continue
                result = self.classifier.classify(tensor)
                det["class_id"]   = result["class_id"]
                det["confidence"] = result["confidence"]
                det["label"]      = SIGN_LABELS.get(result["class_id"], "Unknown")
                det["top_k"]      = result["top_k"]
                classified.append(det)
        else:
            classified = detections

        # 3. Run hazard engine
        alerts = self.hazard.update(classified)

        # 4. Annotate frame
        annotated = TrafficSignDetector.draw_detections(frame_scaled, classified, alerts)
        annotated = self._overlay_hud(annotated, classified, alerts)
        
        # Resize back to original size if needed
        if self.resize_factor < 1.0:
            annotated = cv2.resize(annotated, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

        return annotated, classified, alerts

    # ── HUD overlay ──────────────────────────────────────────────────────────
    def _overlay_hud(self, frame: np.ndarray, detections: list, alerts: list) -> np.ndarray:
        h, w = frame.shape[:2]
        fps_text = f"FPS: {self._stats['fps']:.1f}"
        det_text = f"Signs: {len(detections)}"

        cv2.putText(frame, fps_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, det_text, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Alert banner
        if alerts:
            top = alerts[0]
            level = top["level"]
            color = HazardEngine.hazard_level_color(level)
            banner_h = 48
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - banner_h), (w, h), color, -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(frame, f"  {top['message'].upper()}  — {top['action']}",
                        (10, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)

        return frame

    # ── Main loop (blocking) ──────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(self.cam_index)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {self.cam_index}")

        self.running = True
        reader_thread = threading.Thread(target=self._camera_reader,
                                         args=(cap,), daemon=True)
        reader_thread.start()

        prev_time = time.time()

        print(f"[Pipeline] Speed mode: frame_skip={self.frame_skip}, resize={self.resize_factor:.2f} — press 'q' to quit")
        try:
            while self.running:
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self._frame_count_total += 1
                
                # Skip frames for faster processing on videos
                if self._frame_count_total % self.frame_skip != 0:
                    continue

                annotated, dets, alerts = self._process_frame(frame)

                # FPS calculation
                now = time.time()
                elapsed = now - prev_time
                if elapsed > 0:
                    self._stats["fps"] = 0.9 * self._stats["fps"] + 0.1 * (1.0 / elapsed)
                prev_time = now
                self._stats["detections"]   = len(dets)
                self._stats["frame_count"] += 1
                self._stats["current_dets"] = dets
                self._latest_annotated = annotated

                # Log
                self.logger.log_frame(self._stats["frame_count"],
                                      dets, alerts, self._stats["fps"])

                cv2.imshow("Traffic Sign Recognition", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            self.logger.close()
            print(f"[Pipeline] Stopped - Avg FPS: {self._stats['fps']:.1f}")


    # ── Generator for Flask streaming ─────────────────────────────────────────
    def iter_frames(self):
        """Yields JPEG-encoded bytes of annotated frames for HTTP streaming."""
        cap = cv2.VideoCapture(self.cam_index)
        self.running = True
        frame_count = 0
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Skip frames for faster processing on videos
            if frame_count % self.frame_skip != 0:
                continue
            
            annotated, dets, alerts = self._process_frame(frame)
            self._latest_annotated = annotated
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            yield buf.tobytes()
        cap.release()


    def get_stats(self) -> dict:
        return {**self._stats, "alerts": self.hazard.get_active_alerts()}

    def stop(self):
        self.running = False
