"""FastAPI service that serves the heart disease model.

Endpoints
  GET  /            short service info
  GET  /health      liveness/readiness check (used by Docker + Kubernetes)
  GET  /model-info  which model is loaded and its metrics
  POST /predict     one patient (JSON) -> prediction + confidence
  POST /predict/batch  list of patients
  GET  /metrics     Prometheus metrics (request count, latency, predictions)

Run locally:  uvicorn api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)
from pydantic import BaseModel, ConfigDict, Field

from heart.predict import SAMPLE, load_meta, load_model, predict_records


# ---------------------------------------------------------------- logging
class JsonFormatter(logging.Formatter):
    """One JSON object per line: easy to search in kubectl logs / Loki / ELK."""

    def format(self, record):
        payload = {"time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
                   "level": record.levelname, "msg": record.getMessage()}
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("heart-api")
logger.handlers = [handler]
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))
logger.propagate = False

# ---------------------------------------------------------------- metrics
REQUESTS = Counter("api_requests_total", "HTTP requests",
                   ["method", "endpoint", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency",
                    ["endpoint"], buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])
CONFIDENCE = Histogram("model_prediction_confidence", "Prediction confidence",
                       buckets=(.5, .6, .7, .8, .9, .95, 1.0))
MODEL_LOADED = Gauge("model_loaded", "1 if the model is loaded")


# ---------------------------------------------------------------- schemas
class Patient(BaseModel):
    """One patient record. Codes follow the UCI Heart Disease dataset."""
    model_config = ConfigDict(json_schema_extra={"example": SAMPLE})

    age: float = Field(..., ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(..., description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(..., description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: Literal[1, 2, 3] = Field(..., description="Slope of peak exercise ST")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Literal[3, 6, 7] = Field(..., description="3 normal, 6 fixed, 7 reversible")


class Prediction(BaseModel):
    prediction: int
    label: str
    probability_disease: float
    confidence: float
    model_version: str


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model(os.getenv("MODEL_PATH"))
        MODEL_LOADED.set(1)
        logger.info("model loaded")
    except Exception as exc:  # keep the app up so /health reports the problem
        MODEL_LOADED.set(0)
        logger.error(f"model failed to load: {exc}")
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
META = load_meta()
MODEL_VERSION = META.get("model_version", "unknown")


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed = time.perf_counter() - start
        path = request.url.path
        if path != "/metrics":  # don't count Prometheus scrapes
            REQUESTS.labels(request.method, path, str(status)).inc()
            LATENCY.labels(path).observe(elapsed)
            logger.info("request", extra={"extra_fields": {
                "request_id": request_id, "method": request.method, "path": path,
                "status": status, "latency_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None}})


def _predict(patients: list[Patient]) -> list[Prediction]:
    try:
        results = predict_records([p.model_dump() for p in patients],
                                  model=load_model(os.getenv("MODEL_PATH")))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not loaded")
    for r in results:
        PREDICTIONS.labels(r["label"]).inc()
        CONFIDENCE.observe(r["confidence"])
        logger.info("prediction", extra={"extra_fields": r})
    return [Prediction(**r, model_version=MODEL_VERSION) for r in results]


@app.get("/")
def root():
    return {"service": "heart-disease-risk-api", "version": app.version,
            "docs": "/docs", "predict": "/predict"}


@app.get("/health")
def health():
    ok = MODEL_LOADED._value.get() == 1
    if not ok:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": META.get("model_name"), "version": MODEL_VERSION}


@app.get("/model-info")
def model_info():
    return META


@app.post("/predict", response_model=Prediction)
def predict(patient: Patient):
    return _predict([patient])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(patients: list[Patient]):
    if not 1 <= len(patients) <= 100:
        raise HTTPException(status_code=422, detail="Send between 1 and 100 patients")
    return _predict(patients)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
