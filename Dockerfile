# ─────────────────────────────────────────────────────────────────────────────
# Krishi AI — Crop Disease Detection Service
# Optimised for cloud deployment: CPU-only PyTorch + ONNX inference.
# No CUDA required — the inference engine auto-selects the ONNX backend on CPU.
#
# Build : docker build -t krishi-ai/crop-disease-detector .
# Run   : docker run -p 8000:8000 krishi-ai/crop-disease-detector
# Deploy: push to GitHub → Railway / Render auto-build from this Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ──────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Build deps needed only to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

# Install CPU-only PyTorch first (avoids pulling 4 GB CUDA wheels).
# torch 2.2.x is the last stable release with +cpu wheels on PyPI.
RUN pip install --prefix=/install --no-cache-dir \
        torch==2.2.2+cpu \
        torchvision==0.17.2+cpu \
        --index-url https://download.pytorch.org/whl/cpu

# Install remaining runtime deps (no training / visualisation packages)
COPY requirements-deploy.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements-deploy.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="Krishi AI — Crop Disease Detector"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.description="CPU-optimised ONNX inference service for crop disease detection"

# Minimal runtime system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

WORKDIR /app

# Application source
COPY config.py  ./config.py
COPY src/       ./src/
COPY api/       ./api/
COPY static/    ./static/

# ONNX model + metadata baked into the image so the service starts without
# any external volume mount (required for Railway / Render / Fly.io).
# The large .pth checkpoint is excluded via .dockerignore.
COPY models/crop_disease_model.onnx  ./models/crop_disease_model.onnx
COPY models/class_names.json         ./models/class_names.json
COPY models/temperature.json         ./models/temperature.json

# Writable runtime directories
RUN mkdir -p data logs

# Non-root user for security
RUN useradd --uid 1001 --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Expose the default port; cloud platforms override with $PORT
EXPOSE 8000

# Health check — uses shell form so ${PORT:-8000} is expanded at runtime
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c \
        "import urllib.request, os; \
         urllib.request.urlopen('http://localhost:' + os.environ.get('PORT','8000') + '/api/v1/health')" \
        || exit 1

# Shell form allows ${PORT:-8000} substitution.
# Single worker keeps RSS under 512 MB on free-tier containers.
CMD ["/bin/sh", "-c", \
     "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info"]
