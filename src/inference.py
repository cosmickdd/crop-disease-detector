"""
Inference Engine for Crop Disease Detection
============================================
Provides a high-performance inference service that:
  - Loads and caches the trained model on first call
  - Accepts images as file paths, PIL Images, or raw bytes
  - Preprocesses images identically to the validation pipeline
  - Runs forward pass on the appropriate device (CUDA / CPU)
  - Enriches raw predictions with agronomic knowledge from disease_db
  - Returns a fully structured DiseasePrediction dataclass

Achieves < 1 second per image on GPU, ~200 ms on modern CPU.

Usage:
    engine = InferenceEngine()
    result = engine.predict_from_bytes(image_bytes)
    # or
    result = engine.predict_from_path("field_photo.jpg")
"""

from __future__ import annotations

import io
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
from PIL import Image
from types import SimpleNamespace

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.disease_db import get_disease_info

# ---------------------------------------------------------------------------
# Optional torch / torchvision — only needed for the PyTorch/CUDA backend.
# CPU cloud deployments (Render / Railway) run ONNX Runtime exclusively;
# neither torch nor torchvision needs to be installed in that environment.
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE: bool = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    from src.dataset import get_inference_transform as _get_inference_transform
    _TV_AVAILABLE: bool = True
except ImportError:                     # torchvision not installed
    _TV_AVAILABLE = False
    _get_inference_transform = None     # type: ignore[assignment]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-numpy helpers — used by the ONNX path (zero torch dependency)
# ---------------------------------------------------------------------------

def _softmax_np(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D float32 array."""
    e = np.exp(x - x.max())
    return (e / e.sum()).astype(np.float32)


# ---------------------------------------------------------------------------
# Prediction Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DiseasePrediction:
    """Structured output returned by the inference engine."""
    crop:              str
    disease:           str
    is_healthy:        bool
    confidence:        float           # 0.0 – 1.0
    class_name:        str             # raw model output label
    pathogen:          Optional[str]
    symptoms:          List[str]
    treatment:         List[str]
    prevention:        List[str]
    severity:          str
    top3:              List[Dict[str, Any]] = field(default_factory=list)
    inference_time_ms: float = 0.0
    backend:           str   = "pytorch"
    below_threshold:   bool  = False
    warning:           Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "crop":              self.crop,
            "disease":           self.disease,
            "is_healthy":        self.is_healthy,
            "confidence":        round(self.confidence, 4),
            "class_name":        self.class_name,
            "pathogen":          self.pathogen,
            "symptoms":          self.symptoms,
            "treatment":         self.treatment,
            "prevention":        self.prevention,
            "severity":          self.severity,
            "top3_predictions":  self.top3,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "backend":           self.backend,
            "below_threshold":   self.below_threshold,
            "warning":           self.warning,
        }


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class InferenceEngine:
    """
    Singleton-friendly inference service.

    Supports two backends selected via ``use_onnx``:
      - "auto"    — ONNX Runtime on CPU (2-3× faster), PyTorch on CUDA.
      - "onnx"    — Force ONNX Runtime (raises if onnxruntime / .onnx absent).
      - "pytorch" — Force PyTorch regardless of device.

    Temperature scaling is applied automatically when
    ``models/temperature.json`` exists (run scripts/calibrate_temperature.py).

    Args:
        model_path:     Path to trained .pth checkpoint (PyTorch backend).
        arch:           Architecture name (must match training).
        class_names:    Ordered class labels; falls back to config.CLASS_NAMES.
        device_str:     "auto" | "cuda" | "cpu"
        image_size:     Input spatial resolution (must match training).
        conf_threshold: Confidence below which a low-confidence warning is logged.
        use_onnx:       Backend selector: "auto" | "onnx" | "pytorch".
    """

    def __init__(
        self,
        model_path:     str  | Path          = config.MODEL_SAVE_PATH,
        arch:           str                   = config.MODEL_ARCH,
        class_names:    Optional[List[str]]   = None,
        device_str:     str                   = config.DEVICE,
        image_size:     int                   = config.IMAGE_SIZE,
        conf_threshold: float                 = config.CONFIDENCE_THRESHOLD,
        use_onnx:       str                   = "auto",
    ) -> None:
        self.model_path     = Path(model_path)
        self.arch           = arch
        self.image_size     = image_size
        self.conf_threshold = conf_threshold
        # Plain string device type ("cpu" / "cuda"); torch.device set only if
        # torch is installed so startup never crashes on CPU-only images.
        self._device_str = self._resolve_device_str(device_str)
        self.device = (
            torch.device(self._device_str)
            if _TORCH_AVAILABLE
            else SimpleNamespace(type=self._device_str)
        )
        # torchvision transform — only needed for the PyTorch/CUDA backend.
        self.transform = (
            _get_inference_transform(image_size)
            if _TV_AVAILABLE and _get_inference_transform is not None
            else None
        )

        # Resolve class names
        self.class_names = class_names or self._load_class_names()

        # Temperature scaling (1.0 = identity / no change)
        self.temperature = self._load_temperature()

        # Resolve backend and load accordingly
        self._backend = self._resolve_backend(use_onnx)
        if self._backend == "onnx":
            self._ort_session = self._load_onnx_session()
            self.model = None
        else:
            self.model = self._load_model()
            self._ort_session = None

        logger.info(
            f"InferenceEngine ready — arch={arch}, device={self._device_str}, "
            f"classes={len(self.class_names)}, backend={self._backend}, "
            f"temperature={self.temperature:.4f}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device_str(device_str: str) -> str:
        """Return 'cuda' or 'cpu' as a plain string (no torch required)."""
        if device_str == "auto":
            return "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
        return device_str

    def _load_class_names(self) -> List[str]:
        """Try to load class names persisted by the Trainer, else fall back to config."""
        names_path = config.CLASS_NAMES_PATH
        if names_path.exists():
            with open(names_path) as f:
                names = json.load(f)
            logger.info(f"Class names loaded from {names_path}")
            return names
        logger.warning("class_names.json not found — using config.CLASS_NAMES")
        return config.CLASS_NAMES

    def _load_temperature(self) -> float:
        """Load learned temperature from calibration. Returns 1.0 if not present."""
        temp_path = config.TEMPERATURE_SAVE_PATH
        if temp_path.exists():
            with open(temp_path) as f:
                data = json.load(f)
            T = float(data.get("temperature", 1.0))
            logger.info(f"Temperature scaling loaded: T={T:.4f}")
            return T
        return 1.0

    def _resolve_backend(self, use_onnx: str) -> str:
        """Return 'onnx' or 'pytorch' based on availability and preference."""
        if use_onnx == "pytorch":
            return "pytorch"
        onnx_available = False
        try:
            import onnxruntime  # noqa: F401
            onnx_available = config.ONNX_SAVE_PATH.exists()
        except ImportError:
            pass
        if use_onnx == "onnx":
            if not onnx_available:
                raise RuntimeError(
                    "ONNX backend requested but onnxruntime is not installed "
                    f"or {config.ONNX_SAVE_PATH} does not exist."
                )
            return "onnx"
        # "auto": prefer ONNX on CPU (faster), PyTorch on CUDA
        if onnx_available and self.device.type == "cpu":
            return "onnx"
        return "pytorch"

    def _load_onnx_session(self):
        """Create an ONNX Runtime InferenceSession."""
        import onnxruntime as ort
        onnx_path = config.ONNX_SAVE_PATH
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at '{onnx_path}'. "
                "Run scripts/export_onnx.py to generate it."
            )
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device.type == "cuda"
            else ["CPUExecutionProvider"]
        )
        sess = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(
            f"ONNX Runtime session — path={onnx_path}, "
            f"providers={sess.get_providers()}"
        )
        return sess

    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint or raise a clear error."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at '{self.model_path}'. "
                "Run src/train.py first to train and save the model."
            )
        from src.model import load_checkpoint
        model = load_checkpoint(
            str(self.model_path),
            arch=self.arch,
            num_classes=len(self.class_names),
            device=self.device,
        )
        return model

    # ------------------------------------------------------------------
    # Image loading helpers
    # ------------------------------------------------------------------

    def _bytes_to_pil(self, image_bytes: bytes) -> Image.Image:
        """Convert raw image bytes to a PIL RGB image."""
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise ValueError(
                f"Cannot decode image bytes. Ensure the file is a valid "
                f"JPG, PNG, or WEBP image. Original error: {exc}"
            ) from exc
        return img

    def _pil_to_tensor(self, img: Image.Image) -> torch.Tensor:
        """Apply inference transforms and add a batch dimension."""
        return self.transform(img).unsqueeze(0).to(self.device)

    def _pil_to_numpy(self, img: Image.Image) -> np.ndarray:
        """Preprocess PIL image to float32 (1, C, H, W) for ONNX Runtime.

        Bypasses torchvision.transforms.ToTensor — which calls torch.from_numpy
        internally — so the ONNX path has zero dependency on PyTorch's numpy
        bridge.  Replicates get_val_transforms: Resize → /255 → Normalize.
        """
        # Resize (matches get_val_transforms Resize((image_size, image_size)))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        # PIL RGB uint8 → float32 [0, 1] in HWC layout
        arr = np.array(img, dtype=np.float32) / 255.0        # (H, W, 3)
        # HWC → CHW
        arr = arr.transpose(2, 0, 1)                         # (3, H, W)
        # ImageNet normalisation
        mean = np.array(config.MEAN, dtype=np.float32).reshape(3, 1, 1)
        std  = np.array(config.STD,  dtype=np.float32).reshape(3, 1, 1)
        arr  = (arr - mean) / std
        return arr[np.newaxis].astype(np.float32)            # (1, 3, H, W)

    # ------------------------------------------------------------------
    # Core prediction logic
    # ------------------------------------------------------------------

    def _build_prediction_np(
        self,
        logits_np: np.ndarray,   # shape (num_classes,) float32
        inference_ms: float,
        backend: str,
    ) -> DiseasePrediction:
        """Temperature scaling + softmax + top-k using pure numpy.

        This is the primary prediction builder on CPU deployments — it has
        zero dependency on torch or torchvision.
        """
        probs = _softmax_np(logits_np / self.temperature)

        top_idx    = int(np.argmax(probs))
        confidence = float(probs[top_idx])
        class_name = self.class_names[top_idx]

        k         = min(3, len(self.class_names))
        top3_idxs = np.argsort(probs)[::-1][:k]
        top3 = [
            {
                "class_name": self.class_names[int(i)],
                "confidence": round(float(probs[i]), 4),
            }
            for i in top3_idxs
        ]

        # Uncertainty gate — catches non-plant images and ambiguous predictions.
        margin           = float(probs[top3_idxs[0]] - probs[top3_idxs[1]]) if k > 1 else 1.0
        below_threshold  = confidence < config.CONFIDENCE_WARN_THRESHOLD
        low_margin       = margin < config.MARGIN_WARN_THRESHOLD
        uncertain        = below_threshold or low_margin
        if uncertain:
            if below_threshold:
                warning: Optional[str] = (
                    f"Low confidence ({confidence:.0%}). "
                    "The image may not be a recognisable crop leaf, "
                    "or the plant's condition is outside the model's training "
                    "distribution. Upload a clear, close-up photo of a plant leaf."
                )
            else:
                warning = (
                    f"Ambiguous result — top two classes are very similar "
                    f"(margin {margin:.0%}). Consider re-capturing the leaf "
                    "from a different angle or lighting condition."
                )
        else:
            warning = None

        if confidence < self.conf_threshold:
            logger.warning(
                f"Low confidence prediction: {class_name} ({confidence:.2%}). "
                "Image may not be a crop leaf or outside the training distribution."
            )

        info = get_disease_info(class_name)
        return DiseasePrediction(
            crop=info["crop"],
            disease=info["disease"],
            is_healthy=info["is_healthy"],
            confidence=confidence,
            class_name=class_name,
            pathogen=info.get("pathogen"),
            symptoms=info.get("symptoms", []),
            treatment=info.get("treatment", []),
            prevention=info.get("prevention", []),
            severity=info.get("severity", "Unknown"),
            top3=top3,
            inference_time_ms=inference_ms,
            backend=backend,
            below_threshold=uncertain,
            warning=warning,
        )

    def _build_prediction(
        self,
        logits,          # torch.Tensor, shape (num_classes,) on CPU
        inference_ms: float,
        backend: str,
    ) -> DiseasePrediction:
        """PyTorch-backed prediction builder (CUDA / local use only)."""
        probs = F.softmax(logits / self.temperature, dim=0)

        # Top-1
        top_prob, top_idx = probs.max(dim=0)
        confidence = top_prob.item()
        class_name = self.class_names[int(top_idx.item())]

        # Top-3
        top3_vals, top3_idxs = probs.topk(min(3, len(self.class_names)))
        top3 = [
            {
                "class_name": self.class_names[int(i.item())],
                "confidence": round(v.item(), 4),
            }
            for v, i in zip(top3_vals, top3_idxs)
        ]

        # Uncertainty gate
        margin          = float(top3_vals[0].item() - top3_vals[1].item()) if len(top3_vals) > 1 else 1.0
        below_threshold = confidence < config.CONFIDENCE_WARN_THRESHOLD
        low_margin      = margin < config.MARGIN_WARN_THRESHOLD
        uncertain       = below_threshold or low_margin
        if uncertain:
            warning: Optional[str] = (
                f"Low confidence ({confidence:.0%}). The image may not be a "
                "recognisable crop leaf, or the plant's condition is outside "
                "the model's training distribution."
            ) if below_threshold else (
                f"Ambiguous result — top two classes are very similar "
                f"(margin {margin:.0%}). Consider re-capturing the leaf."
            )
        else:
            warning = None

        if confidence < self.conf_threshold:
            logger.warning(
                f"Low confidence prediction: {class_name} ({confidence:.2%}). "
                "Image may not be a crop leaf or outside the training distribution."
            )

        info = get_disease_info(class_name)
        return DiseasePrediction(
            crop=info["crop"],
            disease=info["disease"],
            is_healthy=info["is_healthy"],
            confidence=confidence,
            class_name=class_name,
            pathogen=info.get("pathogen"),
            symptoms=info.get("symptoms", []),
            treatment=info.get("treatment", []),
            prevention=info.get("prevention", []),
            severity=info.get("severity", "Unknown"),
            top3=top3,
            inference_time_ms=inference_ms,
            backend=backend,
            below_threshold=uncertain,
            warning=warning,
        )

    def _predict_pytorch(self, img: Image.Image) -> DiseasePrediction:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is not installed. "
                "Cannot use the 'pytorch' backend on this machine. "
                "Install torch, or let the engine auto-select the ONNX backend."
            )
        with torch.no_grad():
            t_start = time.perf_counter()
            tensor  = self._pil_to_tensor(img)
            logits  = self.model(tensor).squeeze(0).cpu()   # (num_classes,)
        return self._build_prediction(logits, (time.perf_counter() - t_start) * 1000, "pytorch")

    def _predict_onnx(self, img: Image.Image) -> DiseasePrediction:
        # Pure numpy — no torch involved at all.
        t_start   = time.perf_counter()
        np_input  = self._pil_to_numpy(img)
        logits_np = self._ort_session.run(["logits"], {"input": np_input})[0]  # (1, C)
        return self._build_prediction_np(
            logits_np.squeeze(0),                           # (num_classes,)
            (time.perf_counter() - t_start) * 1000,
            "onnx",
        )

    def _predict_pil(self, img: Image.Image) -> DiseasePrediction:
        img = img.convert("RGB")
        return self._predict_onnx(img) if self._backend == "onnx" else self._predict_pytorch(img)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_bytes(self, image_bytes: bytes) -> DiseasePrediction:
        """Run inference on raw image bytes (JPG / PNG / WEBP)."""
        return self._predict_pil(self._bytes_to_pil(image_bytes))

    def predict_from_pil(self, img: Image.Image) -> DiseasePrediction:
        """Run inference on a PIL Image object."""
        return self._predict_pil(img)

    def predict_from_path(self, image_path: str | Path) -> DiseasePrediction:
        """Run inference from a file path."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        return self._predict_pil(Image.open(path))

    def predict_batch(self, image_bytes_list: List[bytes]) -> List[DiseasePrediction]:
        """Run inference on a list of raw image bytes (sequential)."""
        return [self.predict_from_bytes(b) for b in image_bytes_list]


# ---------------------------------------------------------------------------
# Module-level singleton helper (lazy initialisation)
# ---------------------------------------------------------------------------

_engine_instance: Optional[InferenceEngine] = None


def get_engine() -> InferenceEngine:
    """
    Return the module-level InferenceEngine singleton.
    Creates it on first call.  Raises FileNotFoundError if no model exists.
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = InferenceEngine()
    return _engine_instance
