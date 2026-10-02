# ==============================================================================
# RAIN-X: Regime-Aware Neural Post-Processing Engine (SIH 26080)
# Multi-stage production container for NCMRWF Operational Forecasting Console
# ==============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install essential system dependencies (libgomp1 is required by LightGBM and PyTorch)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY api/ api/
COPY src/ src/
COPY data/ data/
COPY web/ web/
COPY tests/ tests/
COPY train.py .
COPY README.md .

# Generate synoptic dataset and train models if not already generated
RUN python3 data/dataset_generator.py && python3 train.py

# Expose internal API and Web Console port
EXPOSE 8000

# Define container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Launch the FastAPI operational console
CMD ["python3", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
