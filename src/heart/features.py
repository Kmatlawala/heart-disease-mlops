"""Preprocessing: one scikit-learn ColumnTransformer used for training AND serving."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config


def build_preprocessor() -> ColumnTransformer:
    """Numbers: fill gaps with the median, then scale.
    Categories: fill gaps with the most common value, then one-hot encode.
    Unknown category values seen at prediction time are ignored safely.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary",
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, config.NUMERIC_FEATURES),
        ("cat", categorical, config.CATEGORICAL_FEATURES),
    ])


def build_pipeline(model) -> Pipeline:
    """Preprocessing + model in a single object, saved together as one file."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])
