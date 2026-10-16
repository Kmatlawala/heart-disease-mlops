"""Tests for the FastAPI service, using a small model trained on synthetic data."""
import joblib
import pytest
from fastapi.testclient import TestClient

from heart.predict import SAMPLE, load_model


@pytest.fixture
def client(trained_pipeline, tmp_path, monkeypatch):
    path = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, path)
    monkeypatch.setenv("MODEL_PATH", str(path))
    load_model.cache_clear()
    from api.main import app
    with TestClient(app) as c:  # startup loads the model from MODEL_PATH
        yield c
    load_model.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert body["label"] in ("heart disease", "no heart disease")


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_out_of_range_value(client):
    assert client.post("/predict", json=dict(SAMPLE, age=500)).status_code == 422
    assert client.post("/predict", json=dict(SAMPLE, cp=9)).status_code == 422


def test_batch_predict(client):
    r = client.post("/predict/batch", json=[SAMPLE, dict(SAMPLE, age=40)])
    assert r.status_code == 200 and len(r.json()) == 2


def test_metrics_endpoint_counts_predictions(client):
    client.post("/predict", json=SAMPLE)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
