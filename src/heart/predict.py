"""Load the saved pipeline and make predictions (used by the API and scripts).

Run a quick check:  python -m heart.predict
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from heart import config


@lru_cache(maxsize=1)
def load_model(path: str | None = None):
    return joblib.load(Path(path) if path else config.MODEL_PATH)


def load_meta() -> dict:
    if config.MODEL_META_PATH.exists():
        return json.loads(config.MODEL_META_PATH.read_text())
    return {}


def predict_records(records: list[dict], model=None, threshold: float = 0.5) -> list[dict]:
    """Return prediction, label and confidence for each patient record."""
    model = model or load_model()
    X = pd.DataFrame(records)[config.FEATURES]
    proba = model.predict_proba(X)[:, 1]
    out = []
    for p in proba:
        pred = int(p >= threshold)
        out.append({
            "prediction": pred,
            "label": "heart disease" if pred else "no heart disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
        })
    return out


SAMPLE = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}

if __name__ == "__main__":
    print(json.dumps(predict_records([SAMPLE]), indent=2))
