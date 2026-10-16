"""Shared test fixtures. Tests use synthetic data, so they need no network."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.features import build_pipeline


def make_raw(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Fake data in the raw UCI layout (with 'num' 0-4 and some '?' values)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 190, n),
        "chol": rng.integers(130, 400, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(90, 200, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 5, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(object),
        "thal": rng.choice([3, 6, 7], n).astype(object),
    })
    # make the label depend on features so a model can learn something
    risk = (df["age"] > 55).astype(int) + df["exang"] + (df["cp"] == 4).astype(int)
    df["num"] = np.where(risk >= 2, rng.integers(1, 5, n), 0)
    df.loc[:3, "ca"] = "?"
    df.loc[4:5, "thal"] = "?"
    return df


@pytest.fixture
def raw_df():
    return make_raw()


@pytest.fixture
def clean_df(raw_df):
    from heart.data import clean
    return clean(raw_df)


@pytest.fixture
def trained_pipeline(clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    return pipe
