"""
CNN Model Architectures for Crop Disease Detection
====================================================
Supports: MobileNetV3-Large (default), MobileNetV3-Small,
          EfficientNet-B0, ResNet50

All models use transfer learning from ImageNet pretrained weights.
The classifier head is replaced with a new fully-connected layer
sized to the number of disease classes.

Usage:
    from src.model import build_model, load_checkpoint

    model = build_model(arch="mobilenet_v3_large", num_classes=38)
    model = load_checkpoint("models/crop_disease_model.pth",
                            arch="mobilenet_v3_large", num_classes=38, device=device)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from typing import Callable, Dict


# ---------------------------------------------------------------------------
# Internal builders — one per supported architecture
# ---------------------------------------------------------------------------

def _build_mobilenet_v3_large(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _build_mobilenet_v3_small(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def _build_efficientnet_b0(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def _build_resnet50(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


_ARCH_BUILDERS: Dict[str, Callable] = {
    "mobilenet_v3_large": _build_mobilenet_v3_large,
    "mobilenet_v3_small": _build_mobilenet_v3_small,
    "efficientnet_b0":    _build_efficientnet_b0,
    "resnet50":           _build_resnet50,
}


# ---------------------------------------------------------------------------
# Layer freezing helpers
# ---------------------------------------------------------------------------

def _freeze_all_except_classifier(model: nn.Module, arch: str) -> None:
    """
    Freeze every parameter except the classifier head.
    Useful for first-stage training where only the head is updated.
    """
    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze head based on architecture
    if arch in ("mobilenet_v3_large", "mobilenet_v3_small"):
        for param in model.classifier.parameters():  # type: ignore[union-attr]
            param.requires_grad = True
    elif arch == "efficientnet_b0":
        for param in model.classifier.parameters():  # type: ignore[union-attr]
            param.requires_grad = True
    elif arch == "resnet50":
        for param in model.fc.parameters():  # type: ignore[union-attr]
            param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all model parameters for full fine-tuning (stage 2)."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_model(
    arch: str = "mobilenet_v3_large",
    num_classes: int = 38,
    pretrained: bool = True,
    freeze_base: bool = True,
) -> nn.Module:
    """
    Build a crop disease classification model.

    Args:
        arch:         Architecture name (mobilenet_v3_large | mobilenet_v3_small |
                      efficientnet_b0 | resnet50).
        num_classes:  Number of output disease/healthy classes.
        pretrained:   Initialise backbone with ImageNet weights.
        freeze_base:  If True, freeze backbone layers and only train classifier
                      head — recommended for first training stage.

    Returns:
        PyTorch nn.Module ready for training or inference.

    Raises:
        ValueError: If an unsupported architecture name is supplied.
    """
    if arch not in _ARCH_BUILDERS:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Supported: {list(_ARCH_BUILDERS.keys())}"
        )

    model = _ARCH_BUILDERS[arch](num_classes=num_classes, pretrained=pretrained)

    if freeze_base:
        _freeze_all_except_classifier(model, arch)

    return model


def load_checkpoint(
    path: str,
    arch: str,
    num_classes: int,
    device: torch.device,
) -> nn.Module:
    """
    Load a trained model from a checkpoint file.

    Supports both raw state_dict files and full checkpoint dicts
    (containing 'model_state_dict', 'arch', 'num_classes' keys).

    Args:
        path:        Absolute path to the .pth checkpoint file.
        arch:        Architecture that was used during training.
        num_classes: Number of classes the model was trained on.
        device:      torch.device to map the loaded tensors to.

    Returns:
        Model in eval mode on the specified device.
    """
    model = build_model(
        arch=arch,
        num_classes=num_classes,
        pretrained=False,
        freeze_base=False,
    )

    # Allow weights_only=False for full checkpoint dicts
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Handle both wrapped checkpoint dicts and raw state_dicts
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Return total, trainable, and frozen parameter counts."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total":     total,
        "trainable": trainable,
        "frozen":    total - trainable,
    }
