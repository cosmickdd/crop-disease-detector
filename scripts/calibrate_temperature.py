"""
Temperature Scaling Calibration
================================
Fits a single scalar temperature T on the validation set such that:
    calibrated_probs = softmax(logits / T)
minimises Negative Log-Likelihood, improving confidence calibration without
changing prediction accuracy.

Run once after training:
    python scripts/calibrate_temperature.py

Saves models/temperature.json with the optimal T.
Prints ECE (Expected Calibration Error) before and after.

Reference:
    Guo et al. (2017) "On Calibration of Modern Neural Networks"
    https://arxiv.org/abs/1706.04599
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.dataset import get_val_transforms
from src.model import load_checkpoint

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ECE metric
# ---------------------------------------------------------------------------

def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Expected Calibration Error (lower = better; 0.0 = perfectly calibrated).
    Uses equal-width confidence bins over [0, 1].
    """
    confidences, preds = probs.max(dim=1)
    accuracies = preds.eq(labels)
    ece = 0.0
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidences[mask].mean().item()
        avg_acc  = accuracies[mask].float().mean().item()
        ece += mask.float().mean().item() * abs(avg_conf - avg_acc)
    return ece


# ---------------------------------------------------------------------------
# Collect logits from validation set
# ---------------------------------------------------------------------------

def collect_logits(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all raw logits and labels from the validation set."""
    if not config.VAL_DIR.exists():
        raise FileNotFoundError(f"Validation dir not found: {config.VAL_DIR}")

    transform = get_val_transforms(config.IMAGE_SIZE)
    val_ds    = datasets.ImageFolder(str(config.VAL_DIR), transform=transform)
    loader    = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = load_checkpoint(
        str(config.MODEL_SAVE_PATH),
        arch=config.MODEL_ARCH,
        num_classes=config.NUM_CLASSES,
        device=device,
    )
    model.eval()

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    logger.info(f"Collecting logits on {len(val_ds)} validation samples …")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  logits", leave=False):
            images = images.to(device)
            logits = model(images).cpu()
            all_logits.append(logits)
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


# ---------------------------------------------------------------------------
# Temperature optimisation
# ---------------------------------------------------------------------------

def optimise_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Find scalar T* that minimises NLL(softmax(logits / T), labels).
    Coarse grid search [0.1, 5.0] then binary refinement.
    """
    nll = torch.nn.CrossEntropyLoss()

    def nll_at(T: float) -> float:
        return nll(logits / max(T, 1e-6), labels).item()

    # Coarse scan
    best_T   = 1.0
    best_val = nll_at(1.0)
    for t10 in range(1, 51):             # T in {0.1, 0.2, …, 5.0}
        T = t10 / 10.0
        v = nll_at(T)
        if v < best_val:
            best_val = v
            best_T   = T

    # Binary refinement ±0.5 around best_T
    lo = max(0.01, best_T - 0.5)
    hi = best_T + 0.5
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if nll_at(mid - 1e-5) < nll_at(mid + 1e-5):
            hi = mid
        else:
            lo = mid

    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logits, labels = collect_logits(device)

    # Before calibration
    probs_raw  = F.softmax(logits, dim=1)
    ece_before = expected_calibration_error(probs_raw, labels)
    acc        = probs_raw.argmax(dim=1).eq(labels).float().mean().item()
    logger.info(
        f"Before calibration — Accuracy: {acc:.4f}  ECE: {ece_before:.4f}"
    )

    # Optimise
    logger.info("Optimising temperature …")
    T = optimise_temperature(logits, labels)
    logger.info(f"Optimal temperature T = {T:.4f}")

    # After calibration
    probs_cal = F.softmax(logits / T, dim=1)
    ece_after = expected_calibration_error(probs_cal, labels)
    acc_after = probs_cal.argmax(dim=1).eq(labels).float().mean().item()
    logger.info(
        f"After  calibration — Accuracy: {acc_after:.4f}  ECE: {ece_after:.4f}"
    )
    logger.info(f"ECE reduction: {ece_before - ece_after:+.4f} ({ece_before:.4f} → {ece_after:.4f})")

    # Save
    result = {
        "temperature": round(T, 6),
        "ece_before":  round(ece_before, 6),
        "ece_after":   round(ece_after, 6),
        "accuracy":    round(acc_after, 6),
    }
    save_path = config.TEMPERATURE_SAVE_PATH
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved → {save_path}")
    logger.info(
        "Restart the API server so InferenceEngine picks up the new temperature."
    )


if __name__ == "__main__":
    main()
