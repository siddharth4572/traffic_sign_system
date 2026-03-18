"""
models/cnn_classifier.py
ResNet50/EfficientNet/ViT-based traffic sign classifier — training + inference.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time


# ── Model builder ─────────────────────────────────────────────────────────────
def build_classifier(backbone: str = "resnet50",
                     num_classes: int = 43,
                     dropout: float = 0.4,
                     pretrained: bool = True) -> nn.Module:
    """
    Returns a fine-tune-ready classifier.
    Supported backbones: resnet50, efficientnet_b0, vit_b_16
    """
    weights = "IMAGENET1K_V1" if pretrained else None

    if backbone == "resnet50":
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    elif backbone == "vit_b_16":
        model = models.vit_b_16(weights=weights)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    return model


# ── Trainer ───────────────────────────────────────────────────────────────────
class ClassifierTrainer:

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.device = torch.device(cfg["cnn"]["device"]
                                   if torch.cuda.is_available() else "cpu")
        self.num_classes = cfg["dataset"]["num_classes"]
        self.epochs      = cfg["cnn"]["epochs"]
        self.ckpt_dir    = Path(cfg["paths"]["checkpoints"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.model = build_classifier(
            backbone    = cfg["cnn"]["backbone"],
            num_classes = self.num_classes,
            dropout     = cfg["cnn"]["dropout"],
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr           = cfg["cnn"]["learning_rate"],
            weight_decay = cfg["cnn"]["weight_decay"],
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)

    # ── One epoch ──────────────────────────────────────────────────────────────
    def _run_epoch(self, loader, train: bool = True):
        self.model.train(train)
        total_loss, correct, total = 0.0, 0, 0

        with torch.set_grad_enabled(train):
            for imgs, labels in tqdm(loader, leave=False):
                imgs   = imgs.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(imgs)
                loss   = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += imgs.size(0)

        return total_loss / total, correct / total

    # ── Full training loop ────────────────────────────────────────────────────
    def train(self, train_loader, val_loader):
        best_acc = 0.0
        history  = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._run_epoch(train_loader, train=True)
            va_loss, va_acc = self._run_epoch(val_loader,   train=False)
            self.scheduler.step()
            elapsed = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(va_loss)
            history["val_acc"].append(va_acc)

            print(f"Epoch [{epoch:03d}/{self.epochs}] "
                  f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} "
                  f"({elapsed:.1f}s)")

            if va_acc > best_acc:
                best_acc = va_acc
                torch.save(self.model.state_dict(),
                           self.ckpt_dir / "cnn_classifier.pt")
                print(f"  ✓ New best val_acc={best_acc:.4f} — checkpoint saved")

        print(f"\n[Trainer] Done. Best val accuracy: {best_acc:.4f}")
        return history


# ── Inference wrapper ─────────────────────────────────────────────────────────
class TrafficSignClassifier:
    """Thin inference wrapper used by the real-time pipeline."""

    def __init__(self, weights_path: str, cfg: dict):
        self.device = torch.device(
            cfg["cnn"]["device"] if torch.cuda.is_available() else "cpu")
        self.model = build_classifier(
            backbone    = cfg["cnn"]["backbone"],
            num_classes = cfg["dataset"]["num_classes"],
        ).to(self.device)
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device))
        self.model.eval()
        self.top_k = 3

    @torch.no_grad()
    def classify(self, tensor: torch.Tensor) -> dict:
        """
        tensor: (1, 3, H, W) preprocessed crop.
        Returns top-k predictions.
        """
        tensor = tensor.to(self.device)
        logits = self.model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]
        top_probs, top_ids = probs.topk(self.top_k)
        return {
            "class_id":   int(top_ids[0]),
            "confidence": float(top_probs[0]),
            "top_k": [
                {"class_id": int(i), "prob": float(p)}
                for i, p in zip(top_ids, top_probs)
            ],
        }
