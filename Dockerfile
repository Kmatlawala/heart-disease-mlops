# ---- Heart Disease Risk API ------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    MODEL_PATH=/app/models/model.joblib \
    PORT=8000

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what is needed to serve: package code, API and the trained model
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Run as a non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# One worker per container: Kubernetes scales by adding pods (keeps metrics simple)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
