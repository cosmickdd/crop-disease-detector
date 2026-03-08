"""
Training Pipeline for Crop Disease Detection
=============================================
Two-stage transfer learning strategy:
  Stage 1  — Freeze backbone, train classifier head only (fast convergence).
  Stage 2  — Unfreeze all layers and fine-tune end-to-end at lower LR.

Features:
  - Early stopping with configurable patience
  - ReduceLROnPlateau learning rate scheduler
  - Automatic mixed-precision training (AMP) on CUDA
  - Per-epoch metrics: loss, accuracy, precision, recall, F1
  - Best-checkpoint saving (by validation accuracy)
  - Training history export to JSON for later analysis

Usage:
    python -m src.train
    -- or programmatically --
    from src.train import Trainer
    trainer = Trainer()
    trainer.run()
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Adam
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import load_datasets, create_dataloaders
from src.model import build_model, unfreeze_all, count_parameters

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolve device
# ---------------------------------------------------------------------------

def _resolve_device() -> torch.device:
    if config.DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.DEVICE)


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_metrics(
    all_labels: List[int],
    all_preds: List[int],
    num_classes: int,
) -> Dict[str, float]:
    """Compute macro precision, recall, and F1 for multi-class classification."""
    avg = "macro"
    return {
        "precision": float(precision_score(
            all_labels, all_preds, average=avg, zero_division=0
        )),
        "recall": float(recall_score(
            all_labels, all_preds, average=avg, zero_division=0
        )),
        "f1": float(f1_score(
            all_labels, all_preds, average=avg, zero_division=0
        )),
    }


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience   = patience
        self.min_delta  = min_delta
        self.best_score = None
        self.counter    = 0
        self.should_stop = False

    def step(self, metric: float) -> None:
        """Call after each epoch with the monitored metric (higher = better)."""
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score = metric
            self.counter    = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter}/{self.patience}"
            )
            if self.counter >= self.patience:
                self.should_stop = True


# ---------------------------------------------------------------------------
# Core training / evaluation steps
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler],
    num_classes: int,
) -> Dict[str, float]:
    """Run one full training epoch and return aggregated metrics."""
    model.train()
    running_loss    = 0.0
    correct         = 0
    total           = 0
    all_labels: List[int] = []
    all_preds:  List[int] = []

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast("cuda"):
                outputs = model(images)
                loss    = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
        else:
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct      += preds.eq(labels).sum().item()
        total        += images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.3f}")

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    metrics    = _compute_metrics(all_labels, all_preds, num_classes)
    return {"loss": epoch_loss, "accuracy": epoch_acc, **metrics}


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, float]:
    """Run one full evaluation (val / test) epoch and return aggregated metrics."""
    model.eval()
    running_loss    = 0.0
    correct         = 0
    total           = 0
    all_labels: List[int] = []
    all_preds:  List[int] = []

    for images, labels in tqdm(loader, desc="   eval", leave=False, unit="batch"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss    = criterion(outputs, labels)
        preds   = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        correct      += preds.eq(labels).sum().item()
        total        += images.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    metrics    = _compute_metrics(all_labels, all_preds, num_classes)
    return {"loss": epoch_loss, "accuracy": epoch_acc, **metrics}


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the complete two-stage training pipeline.

    Stage 1: Train classifier head only (freeze backbone).
    Stage 2: Unfreeze all layers and fine-tune end-to-end.
    """

    def __init__(
        self,
        arch:          str  = config.MODEL_ARCH,
        num_classes:   int  = config.NUM_CLASSES,
        batch_size:    int  = config.BATCH_SIZE,
        lr:            float = config.LEARNING_RATE,
        weight_decay:  float = config.WEIGHT_DECAY,
        epochs:        int  = config.NUM_EPOCHS,
        patience:      int  = config.EARLY_STOPPING_PATIENCE,
        num_workers:   int  = config.NUM_WORKERS,
        image_size:    int  = config.IMAGE_SIZE,
        save_path:     Path = config.MODEL_SAVE_PATH,
    ) -> None:
        self.arch         = arch
        self.num_classes  = num_classes
        self.batch_size   = batch_size
        self.lr           = lr
        self.weight_decay = weight_decay
        self.epochs       = epochs
        self.patience     = patience
        self.num_workers  = num_workers
        self.image_size   = image_size
        self.save_path    = Path(save_path)
        self.device       = _resolve_device()

        self.history: Dict[str, List] = {
            "train_loss": [], "train_accuracy": [],
            "val_loss":   [], "val_accuracy":   [],
            "train_f1":   [], "val_f1":         [],
        }

        logger.info(f"Device: {self.device}")
        logger.info(f"Architecture: {self.arch}")

    # ------------------------------------------------------------------
    def _save_checkpoint(
        self,
        model: nn.Module,
        val_acc: float,
        epoch: int,
        class_names: List[str],
    ) -> None:
        """Persist the best model checkpoint."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "arch":             self.arch,
                "num_classes":      self.num_classes,
                "class_names":      class_names,
                "val_accuracy":     val_acc,
                "epoch":            epoch,
            },
            self.save_path,
        )
        # Also persist class names as a standalone JSON for the inference service
        names_path = config.CLASS_NAMES_PATH
        names_path.parent.mkdir(parents=True, exist_ok=True)
        with open(names_path, "w") as f:
            json.dump(class_names, f, indent=2)
        logger.info(f"Checkpoint saved → {self.save_path}  (val_acc={val_acc:.4f})")

    # ------------------------------------------------------------------
    def _save_history(self) -> None:
        history_path = config.LOGS_DIR / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Training history saved → {history_path}")

    # ------------------------------------------------------------------
    def _run_stage(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimiser: torch.optim.Optimizer,
        scheduler: ReduceLROnPlateau,
        early_stopping: EarlyStopping,
        criterion: nn.Module,
        scaler: Optional[GradScaler],
        class_names: List[str],
        stage_label: str,
        max_epochs: int,
    ) -> Tuple[nn.Module, float]:
        """Generic training loop shared between Stage 1 and Stage 2."""
        best_val_acc = 0.0

        for epoch in range(1, max_epochs + 1):
            t0 = time.time()

            train_metrics = _train_epoch(
                model, train_loader, criterion, optimiser,
                self.device, scaler, self.num_classes,
            )
            val_metrics = _eval_epoch(
                model, val_loader, criterion,
                self.device, self.num_classes,
            )

            scheduler.step(val_metrics["loss"])
            early_stopping.step(val_metrics["accuracy"])

            elapsed = time.time() - t0
            logger.info(
                f"[{stage_label}] Epoch {epoch:>3}/{max_epochs} "
                f"| Train Loss: {train_metrics['loss']:.4f} "
                f"Acc: {train_metrics['accuracy']:.4f} "
                f"F1: {train_metrics['f1']:.4f} "
                f"| Val Loss: {val_metrics['loss']:.4f} "
                f"Acc: {val_metrics['accuracy']:.4f} "
                f"F1: {val_metrics['f1']:.4f} "
                f"| {elapsed:.1f}s"
            )

            # Persist history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_accuracy"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_f1"].append(val_metrics["f1"])

            # Save best checkpoint
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                self._save_checkpoint(model, best_val_acc, epoch, class_names)

            if early_stopping.should_stop:
                logger.info(f"Early stopping triggered after epoch {epoch}.")
                break

        return model, best_val_acc

    # ------------------------------------------------------------------
    def run(self) -> None:
        """Execute the full two-stage training pipeline."""
        # ── Load data ───────────────────────────────────────────────────
        logger.info("Loading datasets …")
        train_ds, val_ds, test_ds = load_datasets(
            train_dir=config.TRAIN_DIR,
            val_dir=config.VAL_DIR,
            test_dir=config.TEST_DIR,
            image_size=self.image_size,
        )
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ds, val_ds, test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        class_names = train_ds.classes
        logger.info(
            f"Dataset — train: {len(train_ds)}, val: {len(val_ds)}, "
            f"classes: {len(class_names)}"
        )

        # ── Build model ─────────────────────────────────────────────────
        model = build_model(
            arch=self.arch,
            num_classes=len(class_names),
            pretrained=True,
            freeze_base=True,     # Stage 1: head only
        )
        model.to(self.device)

        param_info = count_parameters(model)
        logger.info(
            f"Parameters — total: {param_info['total']:,} "
            f"trainable: {param_info['trainable']:,} "
            f"frozen: {param_info['frozen']:,}"
        )

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler    = GradScaler("cuda") if self.device.type == "cuda" else None

        # ════════════════════════════════════════════════════════════════
        # STAGE 1 — Train classifier head only
        # ════════════════════════════════════════════════════════════════
        s1_epochs = max(5, self.epochs // 3)
        logger.info(
            f"\n{'='*60}\nSTAGE 1 — Classifier head only ({s1_epochs} epochs)\n{'='*60}"
        )
        optimiser_s1 = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr * 5,          # higher LR for head-only stage
            weight_decay=self.weight_decay,
        )
        scheduler_s1  = ReduceLROnPlateau(optimiser_s1, mode="min", factor=0.5, patience=2)
        early_stop_s1 = EarlyStopping(patience=self.patience)

        model, best_s1 = self._run_stage(
            model, train_loader, val_loader,
            optimiser_s1, scheduler_s1, early_stop_s1,
            criterion, scaler, class_names,
            stage_label="S1", max_epochs=s1_epochs,
        )
        logger.info(f"Stage 1 best val accuracy: {best_s1:.4f}")

        # ════════════════════════════════════════════════════════════════
        # STAGE 2 — Fine-tune entire network
        # ════════════════════════════════════════════════════════════════
        remaining_epochs = self.epochs - s1_epochs
        logger.info(
            f"\n{'='*60}\nSTAGE 2 — Full fine-tuning ({remaining_epochs} epochs)\n{'='*60}"
        )
        unfreeze_all(model)

        param_info2 = count_parameters(model)
        logger.info(f"Stage 2 trainable parameters: {param_info2['trainable']:,}")

        optimiser_s2 = Adam(
            model.parameters(),
            lr=self.lr,              # lower LR for full fine-tune
            weight_decay=self.weight_decay,
        )
        scheduler_s2  = ReduceLROnPlateau(optimiser_s2, mode="min", factor=0.5, patience=3)
        early_stop_s2 = EarlyStopping(patience=self.patience)

        model, best_s2 = self._run_stage(
            model, train_loader, val_loader,
            optimiser_s2, scheduler_s2, early_stop_s2,
            criterion, scaler, class_names,
            stage_label="S2", max_epochs=remaining_epochs,
        )
        logger.info(f"Stage 2 best val accuracy: {best_s2:.4f}")

        # ── Test evaluation ─────────────────────────────────────────────
        if test_loader is not None:
            logger.info("\nRunning test set evaluation …")
            # Reload the best checkpoint
            from src.model import load_checkpoint
            model = load_checkpoint(
                str(self.save_path), self.arch,
                len(class_names), self.device,
            )
            test_metrics = _eval_epoch(
                model, test_loader, criterion,
                self.device, len(class_names),
            )
            logger.info(
                f"Test — Loss: {test_metrics['loss']:.4f} "
                f"Acc: {test_metrics['accuracy']:.4f} "
                f"Precision: {test_metrics['precision']:.4f} "
                f"Recall: {test_metrics['recall']:.4f} "
                f"F1: {test_metrics['f1']:.4f}"
            )

        self._save_history()
        logger.info(
            f"\nTraining complete. Best val accuracy: "
            f"{max(best_s1, best_s2):.4f}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = Trainer()
    trainer.run()
