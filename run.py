"""
run.py — Launch the real-time Traffic Sign Recognition pipeline.

Usage:
    python run.py                          # OpenCV window (default)
    python run.py --mode dashboard         # Flask web dashboard
    python run.py --source path/to/video   # Run on a video file
    python run.py --mode dashboard --port 8080
"""

import argparse
import threading
import yaml
import sys
from pathlib import Path


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Sign Recognition & Hazard Prediction System")
    parser.add_argument("--mode",   choices=["window", "dashboard"], default="window",
                        help="Output mode: OpenCV window or web dashboard")
    parser.add_argument("--source", default=None,
                        help="Video file path (default: live camera)")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=5000)
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override camera with file source if provided
    if args.source:
        cfg["inference"]["camera_index"] = args.source

    from modules.pipeline import RealtimePipeline
    pipeline = RealtimePipeline(cfg)

    if args.mode == "dashboard":
        from ui.dashboard import run_dashboard
        # Run pipeline in background thread, dashboard in main thread
        print(f"[Run] Starting dashboard at http://{args.host}:{args.port}")
        run_dashboard(pipeline, host=args.host, port=args.port)
    else:
        print("[Run] Starting OpenCV window (press 'q' to quit)")
        pipeline.run()


if __name__ == "__main__":
    main()
