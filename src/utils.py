"""
Training Visualisation Utilities
==================================
Generates publication-quality plots from training history JSON:
  - Loss curves (train vs validation)
  - Accuracy curves
  - F1 score curves
  - Confusion matrix

Usage:
    python src/utils.py                          # auto-loads logs/training_history.json
    python src/utils.py --history path/to/history.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _setup_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi":     120,
        "font.size":      11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })


def plot_training_curves(
    history: Dict[str, List[float]],
    save_dir: Path | None = None,
) -> None:
    """
    Render and optionally save three side-by-side training curve subplots:
      (1) Loss  (2) Accuracy  (3) F1 Score
    """
    _setup_style()
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Crop Disease Detection — Training History", fontsize=15, y=1.02)

    # ── Loss ────────────────────────────────────────────────────────────
    axes[0].plot(epochs, history["train_loss"], label="Train",      marker="o", linewidth=1.8)
    axes[0].plot(epochs, history["val_loss"],   label="Validation", marker="s", linewidth=1.8)
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── Accuracy ─────────────────────────────────────────────────────────
    axes[1].plot(epochs, [a * 100 for a in history["train_accuracy"]], label="Train",      marker="o", linewidth=1.8)
    axes[1].plot(epochs, [a * 100 for a in history["val_accuracy"]],   label="Validation", marker="s", linewidth=1.8)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 105)
    axes[1].legend()
    axes[1].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # ── F1 Score ──────────────────────────────────────────────────────────
    axes[2].plot(epochs, [f * 100 for f in history["train_f1"]], label="Train",      marker="o", linewidth=1.8)
    axes[2].plot(epochs, [f * 100 for f in history["val_f1"]],   label="Validation", marker="s", linewidth=1.8)
    axes[2].set_title("Macro F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 (%)")
    axes[2].set_ylim(0, 105)
    axes[2].legend()
    axes[2].xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "training_curves.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_dir: Path | None = None,
) -> None:
    """
    Render a normalised confusion matrix heatmap.

    Args:
        cm:           Raw (unnormalised) confusion matrix array shape (C, C).
        class_names:  Ordered list of class labels.
        save_dir:     Optional directory to save the figure.
    """
    _setup_style()

    # Normalise to percentages
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1    # avoid division by zero
    cm_norm = cm_norm / row_sums * 100

    # Shorten class labels for readability
    short_names = [
        n.replace("___", "\n").replace("_", " ")[:30]
        for n in class_names
    ]

    n = len(class_names)
    fig_size = max(12, n * 0.45)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    sns.heatmap(
        cm_norm,
        annot=(n <= 20),          # only annotate cells when ≤ 20 classes
        fmt=".0f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"label": "Row-normalised Recall (%)"},
    )

    ax.set_title("Confusion Matrix (row-normalised)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / "confusion_matrix.png"
        plt.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.show()


def compute_confusion_matrix(
    model_path: str | Path = config.MODEL_SAVE_PATH,
    data_dir:   str | Path = config.VAL_DIR,
) -> tuple:
    """
    Run the validation set through the model and return the confusion matrix
    and class names.

    Returns:
        (numpy confusion matrix, list of class names)
    """
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from sklearn.metrics import confusion_matrix as sk_cm

    import json

    # Resolve device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class names
    names_path = config.CLASS_NAMES_PATH
    if names_path.exists():
        with open(names_path) as f:
            class_names = json.load(f)
    else:
        class_names = config.CLASS_NAMES

    from src.model import load_checkpoint
    from src.dataset import get_val_transforms

    model = load_checkpoint(str(model_path), config.MODEL_ARCH, len(class_names), device)

    ds = datasets.ImageFolder(str(data_dir), transform=get_val_transforms())
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    all_labels, all_preds = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds  = model(images).argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    cm = sk_cm(all_labels, all_preds, labels=list(range(len(class_names))))
    return cm, class_names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise training history")
    parser.add_argument(
        "--history",
        default=str(config.LOGS_DIR / "training_history.json"),
        help="Path to training_history.json generated by train.py",
    )
    parser.add_argument(
        "--save-dir",
        default=str(config.LOGS_DIR),
        help="Directory to save plot images",
    )
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        help="Also compute and plot the confusion matrix on the validation set",
    )
    args = parser.parse_args()

    history_path = Path(args.history)
    save_dir     = Path(args.save_dir)

    if not history_path.exists():
        print(f"ERROR: History file not found: {history_path}")
        print("Train the model first: python -m src.train")
        sys.exit(1)

    with open(history_path) as f:
        history = json.load(f)

    plot_training_curves(history, save_dir=save_dir)

    if args.confusion_matrix:
        cm, class_names = compute_confusion_matrix()
        plot_confusion_matrix(cm, class_names, save_dir=save_dir)
