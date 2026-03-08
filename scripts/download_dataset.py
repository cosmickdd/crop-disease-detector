"""
Dataset Download and Preparation Script
=========================================
Downloads the plant-disease-detection dataset from Kaggle via kagglehub
and organises it into the train / validation / test directory structure
expected by the trainer.

Prerequisites:
    pip install kagglehub
    Place your kaggle.json API key at:
       - Linux/Mac: ~/.kaggle/kaggle.json
       - Windows:   C:\\Users\\<You>\\.kaggle\\kaggle.json

Dataset:
    https://www.kaggle.com/datasets/karagwaanntreasure/plant-disease-detection
    23 disease / healthy classes

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --val-split 0.15 --test-split 0.05
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config


# ---------------------------------------------------------------------------
# Kaggle download
# ---------------------------------------------------------------------------

def download_plantvillage(dest_dir: Path) -> Path:
    """
    Download the plant-disease-detection dataset via kagglehub.
    ``dest_dir`` is accepted for API compatibility but kagglehub manages
    its own cache location.  Returns the path to the Dataset/ sub-folder.
    """
    try:
        import kagglehub
    except ImportError:
        print("ERROR: kagglehub not installed. Run:  pip install kagglehub")
        sys.exit(1)

    print("Downloading karagwaanntreasure/plant-disease-detection via kagglehub …")
    path = kagglehub.dataset_download("karagwaanntreasure/plant-disease-detection")
    source = Path(path) / "Dataset"
    if not source.is_dir():
        return Path(path)
    return source


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_dataset(
    source_dir:  Path,
    output_dir:  Path,
    val_split:   float = 0.15,
    test_split:  float = 0.05,
    seed:        int   = 42,
) -> None:
    """
    Split an ImageFolder dataset into train / validation / test.

    Args:
        source_dir:  Root directory with one sub-folder per class.
        output_dir:  Root to write the split structure into.
        val_split:   Fraction of images per class for validation.
        test_split:  Fraction of images per class for test.
        seed:        Random seed for reproducibility.
    """
    random.seed(seed)

    train_dir = output_dir / "train"
    val_dir   = output_dir / "validation"
    test_dir  = output_dir / "test"

    image_extensions = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
    total_copied = 0

    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} class directories in '{source_dir}'")

    for cls_dir in class_dirs:
        images = [
            f for f in cls_dir.iterdir()
            if f.is_file() and f.suffix in image_extensions
        ]
        if not images:
            print(f"  WARNING: No images found in {cls_dir.name} — skipped")
            continue

        random.shuffle(images)
        n     = len(images)
        n_val  = max(1, int(n * val_split))
        n_test = max(1, int(n * test_split))
        n_train = n - n_val - n_test

        splits = {
            "train":      images[:n_train],
            "validation": images[n_train:n_train + n_val],
            "test":       images[n_train + n_val:],
        }

        for split_name, split_images in splits.items():
            out_cls_dir = output_dir / split_name / cls_dir.name
            out_cls_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, out_cls_dir / img_path.name)
            total_copied += len(split_images)

        print(
            f"  {cls_dir.name:<55} "
            f"train: {n_train:>4}  val: {n_val:>4}  test: {n_test:>4}"
        )

    print(f"\nDone. {total_copied} images copied to '{output_dir}'.")
    print(f"  Train dir:      {train_dir}")
    print(f"  Validation dir: {val_dir}")
    print(f"  Test dir:       {test_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and prepare plant-disease-detection dataset"
    )
    parser.add_argument(
        "--raw-dir",
        default=str(config.DATA_DIR / "raw"),
        help="Directory to download the raw Kaggle zip into",
    )
    parser.add_argument(
        "--output-dir",
        default=str(config.DATASET_DIR),
        help="Output directory for the train/val/test split",
    )
    parser.add_argument(
        "--val-split",
        type=float, default=0.15,
        help="Fraction of images per class to use for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-split",
        type=float, default=0.05,
        help="Fraction of images per class to use for test (default: 0.05)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download — assume raw dataset already exists at --raw-dir",
    )
    args = parser.parse_args()

    raw_dir    = Path(args.raw_dir)
    output_dir = Path(args.output_dir)

    if not args.skip_download:
        source_dir = download_plantvillage(raw_dir)
    else:
        source_dir = raw_dir
        print(f"Skipping download. Using existing raw dataset at: {source_dir}")

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    print(f"\nSplitting dataset → {output_dir}")
    split_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        val_split=args.val_split,
        test_split=args.test_split,
    )


if __name__ == "__main__":
    main()
