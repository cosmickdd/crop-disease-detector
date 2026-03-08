"""
Dataset Loading and Image Preprocessing Pipeline
=================================================
Handles:
  - Train / validation / test dataset loading from ImageFolder structure
  - Data augmentation transforms for training
  - Deterministic resize + normalise transforms for inference
  - DataLoader creation with sensible defaults
  - SingleImageDataset for batch inference on image files

Expected on-disk layout:
  dataset/
    train/
      Apple___Apple_scab/   (image files)
      Tomato___Early_blight/
      ...
    validation/
      Apple___Apple_scab/
      ...
    test/          (optional)
      ...

Classes are inferred automatically from sub-directory names.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Augmentation + normalisation pipeline for the training split.

    Includes:
        - Random resized crop (with scale variation to simulate field distances)
        - Horizontal and vertical flips
        - Random rotation (simulating tilted camera captures)
        - Colour jitter (brightness / contrast / saturation variation)
        - Random perspective distortion (simulates hand-held camera angle)
        - ImageNet normalisation
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic resize + normalise pipeline for validation and test splits.
    Produces exactly the same preprocessing as inference.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """Alias for get_val_transforms — used in the inference pipeline."""
    return get_val_transforms(image_size)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_datasets(
    train_dir: str | Path,
    val_dir: str | Path,
    test_dir: Optional[str | Path] = None,
    image_size: int = 224,
) -> Tuple[
    datasets.ImageFolder,
    datasets.ImageFolder,
    Optional[datasets.ImageFolder],
]:
    """
    Load train / val / test datasets from ImageFolder directory structure.

    Args:
        train_dir:  Path to training split root (contains class sub-directories).
        val_dir:    Path to validation split root.
        test_dir:   Optional path to test split root.
        image_size: Target spatial resolution for resizing.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
        test_dataset is None when test_dir is not provided or doesn't exist.
    """
    train_dataset = datasets.ImageFolder(
        root=str(train_dir),
        transform=get_train_transforms(image_size),
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_dir),
        transform=get_val_transforms(image_size),
    )
    test_dataset: Optional[datasets.ImageFolder] = None
    if test_dir and Path(test_dir).exists():
        test_dataset = datasets.ImageFolder(
            root=str(test_dir),
            transform=get_val_transforms(image_size),
        )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: Optional[datasets.ImageFolder] = None,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Wrap datasets in DataLoaders with optimal settings.

    Args:
        train_dataset: Augmented training dataset.
        val_dataset:   Deterministic validation dataset.
        test_dataset:  Optional test dataset.
        batch_size:    Samples per batch.
        num_workers:   Parallel data loading workers.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
        test_loader is None when test_dataset is None.
    """
    import torch
    _pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=_pin,
        drop_last=True,          # keeps batch sizes uniform for BatchNorm stability
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=_pin,
        persistent_workers=num_workers > 0,
    )
    test_loader: Optional[DataLoader] = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=_pin,
        )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Lightweight dataset for single-image / batch inference on file paths
# ---------------------------------------------------------------------------

class SingleImageDataset(Dataset):
    """
    Minimal Dataset for running inference over a list of image file paths.

    Usage:
        ds = SingleImageDataset(["img1.jpg", "img2.jpg"])
        loader = DataLoader(ds, batch_size=8)
        for tensors, paths in loader:
            predictions = model(tensors)
    """

    def __init__(self, image_paths: List[str], image_size: int = 224) -> None:
        self.image_paths = image_paths
        self.transform   = get_inference_transform(image_size)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path  = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), path
