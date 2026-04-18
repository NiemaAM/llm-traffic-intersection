# ─── Stage 1: builder ────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt


# ─── Stage 2: production image ────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="traffic-intersection-llm"
LABEL description="LLM-Driven Intersection Conflict Resolver API"
LABEL version="1.0.0"

# Non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Create required dirs
RUN mkdir -p data/raw data/processed data/feature_store reports/whylogs mlruns \
 && chown -R appuser:appuser /app

USER appuser

# Environment defaults (override via docker run -e or docker-compose)
ENV PYTHONPATH=/app
ENV MODEL_NAME=gpt-4o-mini
ENV FEW_SHOT=true
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health', timeout=5)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
