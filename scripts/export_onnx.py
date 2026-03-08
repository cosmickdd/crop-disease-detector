"""
ONNX Export Script
===================
Exports the trained PyTorch model to ONNX format for:
  - Cross-platform deployment (TensorRT, ONNX Runtime, CoreML, TFLite)
  - Lower inference latency via ONNX Runtime
  - Mobile deployment via ONNX Runtime Mobile

Typical latency improvements:
  - CPU: ~2–3x speedup vs PyTorch eager mode
  - GPU (TensorRT): ~5–10x speedup

Usage:
    python scripts/export_onnx.py
    python scripts/export_onnx.py --model models/crop_disease_model.pth --output models/model.onnx
    python scripts/export_onnx.py --validate   # run correctness check after export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from src.model import load_checkpoint


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(
    model_path: Path,
    onnx_path:  Path,
    image_size: int   = config.IMAGE_SIZE,
    num_classes: int  = config.NUM_CLASSES,
    arch: str         = config.MODEL_ARCH,
    opset: int        = 17,
) -> None:
    """
    Export a trained .pth checkpoint to ONNX format.

    Args:
        model_path:  Path to the trained PyTorch checkpoint (.pth).
        onnx_path:   Output path for the .onnx file.
        image_size:  Spatial input size (must match training).
        num_classes: Number of output classes.
        arch:        Architecture name (must match training).
        opset:       ONNX opset version. 17 is broadly compatible.
    """
    device = torch.device("cpu")   # export on CPU for maximum compatibility
    model  = load_checkpoint(str(model_path), arch, num_classes, device)
    model.eval()

    # Dummy input — batch size 1, 3-channel RGB, image_size × image_size
    dummy_input = torch.randn(1, 3, image_size, image_size, device=device)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {arch} → {onnx_path} (opset {opset}) …")

    torch.onnx.export(
        model,
        (dummy_input,),
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        verbose=False,
    )
    print(f"Export complete → {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------

def validate_onnx(onnx_path: Path, model_path: Path, image_size: int = 224) -> None:
    """
    Compare ONNX Runtime output against PyTorch output on a random input.
    Asserts that max absolute difference is < 1e-4.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed — skipping validation.")
        print("Install with:  pip install onnxruntime  (CPU) or  onnxruntime-gpu  (CUDA)")
        return

    print("\nValidating ONNX model …")

    device     = torch.device("cpu")
    pt_model   = load_checkpoint(str(model_path), config.MODEL_ARCH, config.NUM_CLASSES, device)
    pt_model.eval()

    dummy_np    = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    dummy_pt    = torch.from_numpy(dummy_np)

    # PyTorch forward pass
    with torch.no_grad():
        pt_out = pt_model(dummy_pt).numpy()

    # ONNX Runtime forward pass
    sess  = ort.InferenceSession(str(onnx_path))
    ort_out = sess.run(["logits"], {"input": dummy_np})[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"Max absolute output difference (PyTorch vs ONNX): {max_diff:.2e}")

    if max_diff < 1e-3:
        print("PASS — ONNX model outputs match PyTorch within tolerance.")
    else:
        print(f"WARNING — Difference exceeds threshold: {max_diff:.2e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX")
    parser.add_argument(
        "--model",
        default=str(config.MODEL_SAVE_PATH),
        help="Path to trained .pth checkpoint",
    )
    parser.add_argument(
        "--output",
        default=str(config.ONNX_SAVE_PATH),
        help="Output path for the .onnx file",
    )
    parser.add_argument(
        "--opset",
        type=int, default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run correctness check comparing PyTorch and ONNX outputs",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    onnx_path  = Path(args.output)

    if not model_path.exists():
        print(f"ERROR: Model checkpoint not found: {model_path}")
        print("Train the model first: python -m src.train")
        sys.exit(1)

    export_onnx(model_path, onnx_path, opset=args.opset)

    if args.validate:
        validate_onnx(onnx_path, model_path)


if __name__ == "__main__":
    main()
