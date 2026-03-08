"""
Krishi AI — Crop Disease Detection API
========================================
FastAPI application entry point.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Or via Python:
    python -m api.main
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from api.routes import router

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: warm up the inference engine at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once at startup so the first request does not incur
    the cold-start loading cost.
    """
    logger.info("Starting up Krishi AI Crop Disease Detection Service …")
    try:
        from src.inference import get_engine
        engine = get_engine()
        logger.info(
            f"Model loaded — arch={engine.arch}, "
            f"device={engine.device}, classes={len(engine.class_names)}"
        )
    except FileNotFoundError as exc:
        logger.warning(
            f"Model checkpoint not found: {exc}. "
            "Service will start in degraded mode — train the model first."
        )
    yield
    logger.info("Shutting down Krishi AI Crop Disease Detection Service.")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Krishi AI — Crop Disease Detection",
    description=(
        "Production-ready REST API for crop disease detection using deep learning.\n\n"
        "**Key capabilities:**\n"
        "- Supports 38 disease / healthy classes across 14 crop types\n"
        "- MobileNetV3-Large backbone for fast mobile inference\n"
        "- Enriched predictions with symptoms, treatment, and prevention advice\n"
        "- Accepts JPG, PNG, and WEBP images up to 10 MB\n\n"
        "**Crops covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, "
        "Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato\n\n"
        "Powered by the PlantVillage dataset and integrated into the Krishi AI platform."
    ),
    version="1.0.0",
    contact={
        "name": "Krishi AI Team",
        "url":  "https://github.com/krishimitraAI",
    },
    license_info={
        "name": "MIT",
    },
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again.",
            "status_code": 500,
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(router, prefix="/api/v1")

# Mount static files (frontend UI)
_static_dir = Path(__file__).resolve().parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Root — serve frontend or JSON fallback
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    index = Path(__file__).resolve().parent.parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {
        "service": "Krishi AI — Crop Disease Detection API",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/health",
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=False,
        workers=1,
        log_level="info",
    )
