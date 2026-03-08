"""
Pydantic schemas for the Crop Disease Detection API.
All request/response models are defined here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class Top3Prediction(BaseModel):
    class_name: str  = Field(..., description="Raw model class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")


class DiseaseDetectionResponse(BaseModel):
    """Structured response returned by POST /detect-disease."""
    crop:              str           = Field(..., description="Detected crop type")
    disease:           str           = Field(..., description="Disease name (or 'Healthy')")
    is_healthy:        bool          = Field(..., description="True if the plant is healthy")
    confidence:        float         = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0–1)")
    class_name:        str           = Field(..., description="Raw model class label (e.g. Tomato___Early_blight)")
    pathogen:          Optional[str] = Field(None, description="Causative organism (fungus / bacterium / virus)")
    symptoms:          List[str]     = Field(default_factory=list, description="Observable disease symptoms")
    treatment:         List[str]     = Field(default_factory=list, description="Recommended treatment steps")
    prevention:        List[str]     = Field(default_factory=list, description="Prevention strategies")
    severity:          str           = Field(..., description="Severity rating of the disease")
    top3_predictions:  List[Top3Prediction] = Field(
        default_factory=list,
        description="Top-3 predictions with confidence scores",
    )
    inference_time_ms: float         = Field(..., description="Server-side inference time in milliseconds")
    backend:           str           = Field("pytorch", description="Inference backend used: pytorch | onnx")
    below_threshold:   bool          = Field(False, description="True if confidence is below warning threshold or prediction is ambiguous")
    warning:           Optional[str] = Field(None, description="Warning message if prediction confidence is low or ambiguous")
    crop_hint_applied: Optional[str] = Field(None, description="Crop hint used to constrain prediction (e.g. 'tomato'), or null for auto-detect")

    model_config = {
        "json_schema_extra": {
            "example": {
                "crop": "Tomato",
                "disease": "Early Blight",
                "is_healthy": False,
                "confidence": 0.93,
                "class_name": "Tomato___Early_blight",
                "pathogen": "Fungus — Alternaria solani",
                "symptoms": [
                    "Brown circular spots with concentric rings",
                    "Yellow halo surrounding lesions",
                    "Lesions first appear on older leaves",
                ],
                "treatment": [
                    "Apply Mancozeb or Chlorothalonil fungicide",
                    "Remove infected lower leaves promptly",
                    "Repeat spray every 7–10 days",
                ],
                "prevention": [
                    "Use drip irrigation to keep foliage dry",
                    "Mulch soil surface to reduce splash dispersal",
                    "Rotate crops for 2–3 years",
                ],
                "severity": "Moderate",
                "top3_predictions": [
                    {"class_name": "Tomato___Early_blight", "confidence": 0.93},
                    {"class_name": "Tomato___Target_Spot",  "confidence": 0.04},
                    {"class_name": "Tomato___healthy",       "confidence": 0.01},
                ],
                "inference_time_ms": 48.7,
            }
        }
    }


class BatchDetectionResponse(BaseModel):
    """Response for POST /batch-detect."""
    results: List[DiseaseDetectionResponse] = Field(
        ..., description="List of predictions — one per uploaded image"
    )
    total_images:     int   = Field(..., description="Number of images processed")
    failed_images:    int   = Field(..., description="Number of images that failed processing")
    total_time_ms:    float = Field(..., description="Total server-side processing time in milliseconds")


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status:       str  = Field(..., description="Service status: ok | degraded | error")
    model_loaded: bool = Field(..., description="Whether the model is loaded and ready")
    device:       str  = Field(..., description="Compute device: cuda | cpu")
    arch:         str  = Field(..., description="Model architecture in use")
    num_classes:  int  = Field(..., description="Number of disease classes")
    version:      str  = Field("1.0.0", description="API version")


class ClassListResponse(BaseModel):
    """Response for GET /classes."""
    classes:     List[str] = Field(..., description="All supported disease class names")
    num_classes: int        = Field(..., description="Total number of classes")


class ErrorResponse(BaseModel):
    """Standardised error envelope."""
    error:   str = Field(..., description="Error type")
    detail:  str = Field(..., description="Detailed error message")
    status_code: int = Field(..., description="HTTP status code")
