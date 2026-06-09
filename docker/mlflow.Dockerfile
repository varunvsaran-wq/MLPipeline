# MLflow tracking server image (backend store = Postgres).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir \
        "mlflow>=2.13,<3.0" \
        "psycopg2-binary>=2.9" \
        "boto3>=1.34"

EXPOSE 5000

# Default command is overridden by docker-compose, but keep a sane fallback.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
