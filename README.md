# Krishi AI — Crop Disease Detection System

Production-ready deep learning pipeline that analyses crop leaf or plant images and returns structured disease diagnoses with agronomic recommendations.

---

## Features

- **38 disease classes** across 14 crop types (PlantVillage dataset)
- **MobileNetV3-Large** backbone — optimised for mobile and edge inference
- **Two-stage transfer learning** — fast convergence with high accuracy (> 90% target)
- **Structured predictions** — disease name, confidence, symptoms, treatment, prevention
- **FastAPI REST API** — single image, batch inference, health check, knowledge base
- **Docker ready** — multi-stage image for minimal container size
- **ONNX export** — for TensorRT, ONNX Runtime, and cross-platform deployment

---

## Supported Crops and Diseases

| Crop | Diseases |
|---|---|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Blueberry | Healthy |
| Cherry | Powdery Mildew, Healthy |
| Corn (Maize) | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot, Healthy |
| Bell Pepper | Bacterial Spot, Healthy |
| Potato | Early Blight, Late Blight, Healthy |
| Raspberry | Healthy |
| Soybean | Healthy |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch, Healthy |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |

---

## Project Structure

```
crop-disease-detector/
├── config.py                   # All hyperparameters and paths
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .gitignore
│
├── src/
│   ├── model.py                # CNN architectures (MobileNetV3, EfficientNet, ResNet50)
│   ├── dataset.py              # Dataset loading and augmentation transforms
│   ├── train.py                # Two-stage training pipeline
│   ├── inference.py            # Inference engine with disease knowledge enrichment
│   ├── disease_db.py           # Agronomic knowledge database (38 classes)
│   └── utils.py                # Training visualisation utilities
│
├── api/
│   ├── main.py                 # FastAPI application with lifespan model warm-up
│   ├── routes.py               # Route handlers (detect, batch, health, classes)
│   └── schemas.py              # Pydantic request/response models
│
├── scripts/
│   ├── download_dataset.py     # Kaggle dataset download + train/val/test split
│   └── export_onnx.py          # ONNX export with correctness validation
│
├── data/
│   └── dataset/
│       ├── train/
│       ├── validation/
│       └── test/
│
├── models/                     # Saved checkpoints (.pth) and ONNX models
└── logs/                       # Training history JSON and visualisation PNGs
```

---

## Quick Start

### 1. Install dependencies

```bash
cd crop-disease-detector
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Download the dataset

Set up your Kaggle API key at `~/.kaggle/kaggle.json`, then:

```bash
python scripts/download_dataset.py
```

This downloads the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) and splits it into `data/dataset/train`, `data/dataset/validation`, and `data/dataset/test`.

### 3. Train the model

```bash
python -m src.train
```

Training uses a two-stage approach:
- **Stage 1** (~8 epochs) — freeze backbone, train classifier head at high LR
- **Stage 2** (~17 epochs) — unfreeze all layers and fine-tune end-to-end

The best checkpoint is saved to `models/crop_disease_model.pth`.

Configuration options (via environment variables or `config.py`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_ARCH` | `mobilenet_v3_large` | Architecture |
| `BATCH_SIZE` | `32` | Training batch size |
| `LEARNING_RATE` | `0.0001` | Base learning rate (Stage 2) |
| `NUM_EPOCHS` | `25` | Total epochs across both stages |
| `DEVICE` | `auto` | `auto` / `cuda` / `cpu` |

### 4. Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## API Reference

### POST `/api/v1/detect-disease`

Detect disease from a single uploaded image.

**Request:** `multipart/form-data` with `image` field (JPG / PNG / WEBP, max 10 MB)

**Response:**
```json
{
  "crop": "Tomato",
  "disease": "Early Blight",
  "is_healthy": false,
  "confidence": 0.93,
  "class_name": "Tomato___Early_blight",
  "pathogen": "Fungus — Alternaria solani",
  "symptoms": [
    "Brown circular spots with concentric rings (target-board pattern)",
    "Yellow halo surrounding lesions",
    "Lesions first appear on older, lower leaves"
  ],
  "treatment": [
    "Apply Mancozeb, Chlorothalonil, or Azoxystrobin fungicide",
    "Begin application at first symptom appearance",
    "Repeat every 7–10 days"
  ],
  "prevention": [
    "Use drip irrigation to keep foliage dry",
    "Mulch soil surface to reduce splash dispersal",
    "Rotate crops for 2–3 years"
  ],
  "severity": "Moderate",
  "top3_predictions": [
    {"class_name": "Tomato___Early_blight", "confidence": 0.93},
    {"class_name": "Tomato___Target_Spot",  "confidence": 0.04},
    {"class_name": "Tomato___healthy",       "confidence": 0.01}
  ],
  "inference_time_ms": 48.7
}
```

### POST `/api/v1/batch-detect`

Process up to 10 images in a single request.

### GET `/api/v1/health`

Service health check — returns model status, device, architecture.

### GET `/api/v1/classes`

Returns all 38 supported disease class names.

### GET `/api/v1/disease/{class_name}`

Fetch disease knowledge base entry without running inference.

Example: `GET /api/v1/disease/Potato___Late_blight`

---

## Deployment

### Cloud deployment (no local Docker required)

Push the repo to GitHub and connect it to **Railway** or **Render** — both build
the Docker image automatically and inject the `$PORT` environment variable.

| Platform | Config file | Free tier |
|---|---|---|
| [Railway](https://railway.app) | `railway.toml` | $5/month credit |
| [Render](https://render.com) | `render.yaml` | 512 MB, sleeps after 15 min |

Steps for Railway:
1. `git push` (models/\*.onnx and metadata are now tracked in git)
2. New Project → Deploy from GitHub repo
3. Railway detects `railway.toml` and builds from `Dockerfile` automatically

Steps for Render:
1. `git push`
2. New Web Service → connect repo → Render detects `render.yaml` automatically

The service uses **ONNX Runtime + CPU-only PyTorch**, so no GPU is required and
the image stays well under 512 MB RSS.

### Local Docker

```bash
docker compose up --build
```

The API will be available at http://localhost:8000.

### Production notes

- The ONNX model (16.9 MB) is baked into the Docker image — no volume mount needed.
- For GPU inference locally, set `DEVICE=cuda` in `docker-compose.yml`.
- Restrict `allow_origins` in `api/main.py` to your specific domain in production.

---

## ONNX Export

Export the trained model for faster cross-platform inference:

```bash
python scripts/export_onnx.py --validate
```

This generates `models/crop_disease_model.onnx` and runs a correctness check comparing ONNX Runtime output against PyTorch output.

---

## Training Visualisation

After training, generate loss / accuracy / F1 plots and a confusion matrix:

```bash
# Training curves only
python src/utils.py

# Training curves + confusion matrix on validation set
python src/utils.py --confusion-matrix
```

Plots are saved to `logs/training_curves.png` and `logs/confusion_matrix.png`.

---

## Performance Targets

| Metric | Target |
|---|---|
| Validation accuracy | > 90% |
| Inference time (CPU) | < 200 ms |
| Inference time (GPU) | < 50 ms |
| Model size (MobileNetV3-Large) | ~17 MB |
| ONNX model size (quantised) | ~5 MB |

---

## Architecture Decision

**MobileNetV3-Large** was chosen as the default because it provides the best balance of:
- Speed: designed for mobile / edge deployment
- Accuracy: comparable to ResNet50 on ImageNet with 3× fewer FLOPs
- Size: ~17 MB — practical for mobile app bundling

To switch architecture, set `MODEL_ARCH=efficientnet_b0` or `MODEL_ARCH=resnet50` in the environment.

---

## Krishi AI Platform Integration

This service is designed to be called from the Krishi AI mobile app or backend:

```python
import requests

response = requests.post(
    "http://your-server/api/v1/detect-disease",
    files={"image": open("field_photo.jpg", "rb")},
)
print(response.json())
```

---

## Future Improvements

- Multi-disease detection per image
- Pest and insect detection
- Nutrient deficiency detection
- Disease severity estimation (mild / moderate / severe)
- Integration with weather forecast APIs for disease risk alerts
- On-device inference via TFLite / ONNX Runtime Mobile
