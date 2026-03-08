"""
Central configuration for the Krishi AI Crop Disease Detection System.

All paths, hyperparameters, and tunable settings live here.
Override values via environment variables for production deployments.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parent
DATA_DIR  = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR  = ROOT_DIR / "logs"

# Create directories at import time (safe for containers)
for _d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# Dataset directory structure expected by ImageFolder
DATASET_DIR = DATA_DIR / "dataset"
TRAIN_DIR   = DATASET_DIR / "train"
VAL_DIR     = DATASET_DIR / "validation"
TEST_DIR    = DATASET_DIR / "test"

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# Options: mobilenet_v3_large | mobilenet_v3_small | efficientnet_b0 | resnet50
MODEL_ARCH  = os.getenv("MODEL_ARCH", "mobilenet_v3_large")
IMAGE_SIZE  = 224           # input spatial resolution
NUM_CLASSES = 23            # 23 disease / healthy classes

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE               = int(os.getenv("BATCH_SIZE", 32))
LEARNING_RATE            = float(os.getenv("LEARNING_RATE", 1e-4))
NUM_EPOCHS               = int(os.getenv("NUM_EPOCHS", 25))
WEIGHT_DECAY             = float(os.getenv("WEIGHT_DECAY", 1e-4))
EARLY_STOPPING_PATIENCE  = int(os.getenv("EARLY_STOPPING_PATIENCE", 5))
NUM_WORKERS              = int(os.getenv("NUM_WORKERS", 0))

# ---------------------------------------------------------------------------
# ImageNet normalisation constants
# ---------------------------------------------------------------------------
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Saved artifacts
# ---------------------------------------------------------------------------
MODEL_SAVE_PATH       = MODELS_DIR / "crop_disease_model.pth"
ONNX_SAVE_PATH        = MODELS_DIR / "crop_disease_model.onnx"
CLASS_NAMES_PATH      = MODELS_DIR / "class_names.json"
TEMPERATURE_SAVE_PATH = MODELS_DIR / "temperature.json"

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.50))
# Predictions below this level get a warning flag sent to the client.
# Catches non-plant images and ambiguous cases (model unsure between classes).
CONFIDENCE_WARN_THRESHOLD = float(os.getenv("CONFIDENCE_WARN_THRESHOLD", 0.75))
# If (top-1 conf - top-2 conf) < this, model is choosing between similar classes.
MARGIN_WARN_THRESHOLD = float(os.getenv("MARGIN_WARN_THRESHOLD", 0.20))
# "auto" → use CUDA if available, else CPU
DEVICE = os.getenv("DEVICE", "auto")

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
API_HOST              = os.getenv("API_HOST", "0.0.0.0")
# Railway / Render inject $PORT; fall back to $API_PORT, then 8000
API_PORT              = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
MAX_IMAGE_SIZE_BYTES  = 10 * 1024 * 1024          # 10 MB hard limit
ALLOWED_EXTENSIONS    = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Dataset class names — 23 classes (alphabetical order matching folder names)
# Dataset: karagwaanntreasure/plant-disease-detection
# ---------------------------------------------------------------------------
CLASS_NAMES: list[str] = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]
