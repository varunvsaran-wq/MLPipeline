# Streamlit ops dashboard (Phase 5).
# Read-only view over the model bundle, the prediction log, and the drift
# signals — deliberately separate from the prediction API so operating the
# system never competes with serving it.

# ---- builder: install deps into an isolated venv --------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && pip install \
        "streamlit>=1.36" \
        "plotly>=5.22" \
        "evidently>=0.4.30" \
        "sqlalchemy>=2.0,<2.1" \
        "pydantic>=2.7,<3.0" \
        "pyyaml>=6.0" \
        "pandas>=2.2,<3.0" \
        "numpy>=1.26,<2.0" \
        "scikit-learn>=1.5,<2.0" \
        "lightgbm>=4.3" \
        "holidays>=0.50" \
        "joblib>=1.4"

# ---- final ----------------------------------------------------------------
FROM python:3.11-slim

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    SERVING_DATASET=avocado

# LightGBM's Linux wheel needs the OpenMP runtime, which python:*-slim leaves out;
# without it `import lightgbm` fails and the app never starts.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r app && useradd -r -g app app
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

COPY config ./config
COPY data ./data
COPY features ./features
COPY models ./models
COPY serving ./serving
COPY monitoring ./monitoring
COPY dashboard ./dashboard

RUN mkdir -p /app/state && chown -R app:app /app/state
ENV SERVING_DB_URI="sqlite:////app/state/predictions.db"

USER app
EXPOSE 8501

HEALTHCHECK --interval=15s --timeout=5s --start-period=25s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
