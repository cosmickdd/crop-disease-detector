"""
API Route Handlers for Crop Disease Detection
==============================================
Endpoints:
  POST /detect-disease         — single image inference
  POST /batch-detect           — multi-image inference (up to 10 images)
  GET  /health                 — service health check
  GET  /classes                — list all supported disease classes
  GET  /disease/{class_name}   — fetch disease info without running inference
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from api.schemas import (
    BatchDetectionResponse,
    ClassListResponse,
    DiseaseDetectionResponse,
    HealthResponse,
    Top3Prediction,
)
from src.disease_db import get_disease_info, list_all_diseases
from src.inference import get_engine, CROP_KEYS

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

_ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp",
}
_MAX_BATCH_SIZE = 10


def _validate_image_upload(file: UploadFile) -> None:
    """Raise HTTP 400 / 413 for invalid uploads."""
    if file.content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported media type '{file.content_type}'. "
                f"Allowed: {sorted(_ALLOWED_CONTENT_TYPES)}"
            ),
        )


def _build_response(prediction) -> DiseaseDetectionResponse:
    """Convert a DiseasePrediction dataclass to the API response schema."""
    return DiseaseDetectionResponse(
        crop=prediction.crop,
        disease=prediction.disease,
        is_healthy=prediction.is_healthy,
        confidence=prediction.confidence,
        class_name=prediction.class_name,
        pathogen=prediction.pathogen,
        symptoms=prediction.symptoms,
        treatment=prediction.treatment,
        prevention=prediction.prevention,
        severity=prediction.severity,
        top3_predictions=[
            Top3Prediction(
                class_name=t["class_name"],
                confidence=t["confidence"],
            )
            for t in prediction.top3
        ],
        inference_time_ms=prediction.inference_time_ms,
        backend=prediction.backend,
        below_threshold=prediction.below_threshold,
        warning=prediction.warning,
        crop_hint_applied=prediction.crop_hint_applied,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/detect-disease",
    response_model=DiseaseDetectionResponse,
    summary="Detect crop disease from a single image",
    description=(
        "Upload a crop leaf or plant image (JPG / PNG / WEBP). "
        "The API returns the predicted disease, confidence score, "
        "symptoms, and recommended treatment."
    ),
    tags=["Inference"],
)
async def detect_disease(
    image: UploadFile = File(
        ...,
        description="Crop leaf / plant image (JPG, PNG, WEBP — max 10 MB)",
    ),
    crop_hint: str = Query(
        ...,
        description=(
            "Crop type to constrain prediction. "
            "Accepted values: apple, corn, pepper, potato, tomato."
        ),
    ),
) -> DiseaseDetectionResponse:
    _validate_image_upload(image)

    image_bytes = await image.read()

    if len(image_bytes) > config.MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image size {len(image_bytes) / 1_048_576:.1f} MB "
                f"exceeds the 10 MB limit."
            ),
        )

    try:
        engine     = get_engine()
        prediction = engine.predict_from_bytes(image_bytes, crop_hint=crop_hint)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Image processing failed: {exc}",
        ) from exc

    logger.info(
        f"detect-disease | {image.filename} → {prediction.class_name} "
        f"({prediction.confidence:.2%}) in {prediction.inference_time_ms:.1f} ms"
    )
    return _build_response(prediction)


@router.post(
    "/batch-detect",
    response_model=BatchDetectionResponse,
    summary="Detect crop diseases from multiple images",
    description=(
        "Upload up to 10 crop images in a single request. "
        "Returns one prediction per image in the same order."
    ),
    tags=["Inference"],
)
async def batch_detect(
    images: List[UploadFile] = File(
        ...,
        description="List of crop images (max 10 per request, each max 10 MB)",
    ),
) -> BatchDetectionResponse:
    if len(images) > _MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Maximum {_MAX_BATCH_SIZE} images per batch. Received: {len(images)}",
        )

    t_start  = time.perf_counter()
    results  = []
    failures = 0

    engine = get_engine()

    for upload in images:
        _validate_image_upload(upload)
        image_bytes = await upload.read()

        if len(image_bytes) > config.MAX_IMAGE_SIZE_BYTES:
            failures += 1
            logger.warning(f"batch-detect | {upload.filename} skipped — exceeds size limit")
            continue

        try:
            prediction = engine.predict_from_bytes(image_bytes)
            results.append(_build_response(prediction))
        except Exception as exc:
            failures += 1
            logger.error(f"batch-detect | {upload.filename} failed: {exc}")

    total_ms = (time.perf_counter() - t_start) * 1000

    return BatchDetectionResponse(
        results=results,
        total_images=len(images),
        failed_images=failures,
        total_time_ms=round(total_ms, 2),
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    try:
        engine = get_engine()
        model_loaded = True
    except FileNotFoundError:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        device=str(get_engine().device) if model_loaded else "none",
        arch=config.MODEL_ARCH,
        num_classes=config.NUM_CLASSES,
        version="1.0.0",
    )


@router.get(
    "/crops",
    summary="List supported crop types for crop_hint",
    tags=["System"],
)
async def list_crops():
    """Returns the crop keys accepted by the crop_hint query parameter."""
    return {"crops": CROP_KEYS}


@router.get(
    "/classes",
    response_model=ClassListResponse,
    summary="List all supported disease classes",
    tags=["System"],
)
async def list_classes() -> ClassListResponse:
    classes = list_all_diseases()
    return ClassListResponse(classes=classes, num_classes=len(classes))


@router.get(
    "/disease/{class_name:path}",
    summary="Fetch disease information by class name",
    description=(
        "Returns detailed agronomic information for a disease class "
        "without running model inference. "
        "Example: /disease/Tomato___Early_blight"
    ),
    tags=["Knowledge Base"],
)
async def get_disease_detail(class_name: str):
    info = get_disease_info(class_name)
    return {"class_name": class_name, **info}
