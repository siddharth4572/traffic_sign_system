"""
train.py — Entry point for training YOLO detector and CNN classifier.

Usage:
    python train.py --mode yolo   --config configs/config.yaml
    python train.py --mode cnn    --config configs/config.yaml
    python train.py --mode all    --config configs/config.yaml
"""

import argparse
import yaml
import sys
import torch

from modules.detector      import TrafficSignDetector
from models.cnn_classifier import ClassifierTrainer
from modules.preprocessor  import build_dataloaders, load_config
from utils.evaluate        import evaluate_classifier


def train_yolo(cfg: dict):
    print("\n" + "="*55)
    print("  PHASE 1 — YOLO DETECTION TRAINING")
    print("="*55)
    detector = TrafficSignDetector(cfg=cfg)
    # Requires a prepared YOLO-format dataset at data/yolo_data.yaml
    detector.train(
        data_yaml  = "data/yolo_data.yaml",
        output_dir = cfg["paths"]["checkpoints"],
    )


def train_cnn(cfg: dict):
    print("\n" + "="*55)
    print("  PHASE 2 — CNN CLASSIFIER TRAINING")
    print("="*55)
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = ClassifierTrainer(cfg)
    history = trainer.train(train_loader, val_loader)

    print("\n[Train] Evaluating best checkpoint …")
    evaluate_classifier(cfg, val_loader)
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["yolo", "cnn", "all"], default="all")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"\n[Train] Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"[Train] Mode: {args.mode}")

    if args.mode in ("yolo", "all"):
        train_yolo(cfg)

    if args.mode in ("cnn", "all"):
        train_cnn(cfg)

    print("\n✓ Training complete. Checkpoints saved to:", cfg["paths"]["checkpoints"])


if __name__ == "__main__":
    main()
