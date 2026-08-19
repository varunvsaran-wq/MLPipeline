# Multi-stage build for the FastAPI serving layer (Phase 3).
# Only the lean serving/inference deps are installed (no prophet/mlflow/dvc), so
# the final image stays small. The model bundle is NOT baked in by default — it
# is mounted at runtime (see docker-compose) so the image is data-free.

# ---- builder: install deps into an isolated venv --------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Explicit, minimal runtime set (inference only — mirrors the [serving] extra
# plus lightgbm/joblib needed to load the bundle).
RUN pip install --upgrade pip && pip install \
        "fastapi>=0.111" \
        "uvicorn[standard]>=0.30" \
        "prometheus-client>=0.20" \
        "sqlalchemy>=2.0" \
        "pydantic>=2.7,<3.0" \
        "pyyaml>=6.0" \
        "pandas>=2.2,<3.0" \
        "numpy>=1.26,<2.0" \
        "scikit-learn>=1.5,<2.0" \
        "lightgbm>=4.3" \
        "holidays>=0.50" \
        "joblib>=1.4"

# ---- final: slim runtime image -------------------------------------------
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SERVING_DATASET=avocado

RUN groupadd -r app && useradd -r -g app app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

# Application code only (the plumbing needed to load a bundle and serve).
COPY config ./config
COPY data ./data
COPY features ./features
COPY models ./models
COPY serving ./serving

# Writable state dir for the SQLite prediction log (a named volume mounts here;
# it inherits this ownership on first creation so the non-root user can write).
RUN mkdir -p /app/state && chown -R app:app /app/state
ENV SERVING_DB_URI="sqlite:////app/state/predictions.db"

USER app
EXPOSE 8000

# Liveness probe — matches the /health acceptance check.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
