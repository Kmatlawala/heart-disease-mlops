"""Download, load and clean the UCI Heart Disease (Cleveland) dataset.

Run as a script:  python -m heart.data
It writes data/raw/heart_raw.csv and data/processed/heart_clean.csv.
"""
from __future__ import annotations

import io
import logging
import urllib.request

import numpy as np
import pandas as pd

from heart import config

log = logging.getLogger(__name__)

# Text labels used by the mirror file -> numeric UCI codes
_CP_MAP = {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4}
_THAL_MAP = {"normal": 3, "fixed": 6, "reversable": 7}
_MIRROR_RENAME = {
    "Age": "age", "Sex": "sex", "ChestPain": "cp", "RestBP": "trestbps",
    "Chol": "chol", "Fbs": "fbs", "RestECG": "restecg", "MaxHR": "thalach",
    "ExAng": "exang", "Oldpeak": "oldpeak", "Slope": "slope", "Ca": "ca",
    "Thal": "thal", "AHD": "num",
}


def _fetch(url: str, timeout: int = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def parse_uci(text: str) -> pd.DataFrame:
    """Parse the official comma-separated UCI file ('?' marks missing values)."""
    return pd.read_csv(io.StringIO(text), header=None, names=config.UCI_COLUMNS,
                       na_values="?")


def parse_mirror(text: str) -> pd.DataFrame:
    """Parse the mirror CSV and convert it to the same layout as the UCI file."""
    df = pd.read_csv(io.StringIO(text), index_col=0).rename(columns=_MIRROR_RENAME)
    df["cp"] = df["cp"].map(_CP_MAP)
    df["thal"] = df["thal"].map(_THAL_MAP)
    df["num"] = (df["num"] == "Yes").astype(int)
    return df[config.UCI_COLUMNS]


def download_raw() -> pd.DataFrame:
    """Download from UCI; fall back to the mirror if UCI is unreachable."""
    try:
        df = parse_uci(_fetch(config.UCI_URL))
        log.info("Downloaded %d rows from UCI", len(df))
    except Exception as exc:  # network errors, blocked host, etc.
        log.warning("UCI download failed (%s); using mirror", exc)
        df = parse_mirror(_fetch(config.MIRROR_URL))
        log.info("Downloaded %d rows from mirror", len(df))
    config.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RAW_PATH, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data.

    * make the target binary (UCI 'num' is 0-4; anything > 0 means disease)
    * coerce every column to numbers ('?' or junk -> NaN)
    * set impossible values (outside medical ranges / unknown codes) to NaN
    * drop exact duplicates and rows without a target

    Missing values are NOT filled here on purpose: filling happens inside the
    model pipeline, so the exact same logic runs during training and serving.
    """
    df = df.copy()
    if "num" in df.columns:
        df[config.TARGET] = (pd.to_numeric(df["num"], errors="coerce") > 0).astype(int)
        df = df.drop(columns="num")

    for col in config.FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col, codes in config.VALID_CODES.items():
        df.loc[~df[col].isin(codes) & df[col].notna(), col] = np.nan
    for col, (lo, hi) in config.VALID_RANGES.items():
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    df = df.dropna(subset=[config.TARGET]).drop_duplicates().reset_index(drop=True)
    return df[config.FEATURES + [config.TARGET]]


def load_clean() -> pd.DataFrame:
    """Load the cleaned dataset, creating it first if needed."""
    if not config.CLEAN_PATH.exists():
        build_dataset()
    return pd.read_csv(config.CLEAN_PATH)


def build_dataset() -> pd.DataFrame:
    raw = pd.read_csv(config.RAW_PATH) if config.RAW_PATH.exists() else download_raw()
    df = clean(raw)
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CLEAN_PATH, index=False)
    log.info("Clean data: %d rows, missing values per column: %s",
             len(df), df.isna().sum()[df.isna().sum() > 0].to_dict())
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if config.RAW_PATH.exists():
        config.RAW_PATH.unlink()  # always fetch fresh when run as a script
    build_dataset()
