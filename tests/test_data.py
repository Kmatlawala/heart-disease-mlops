"""Tests for downloading, parsing and cleaning data."""
import numpy as np
import pandas as pd

from heart import config
from heart.data import clean, parse_mirror, parse_uci


def test_clean_makes_binary_target(clean_df):
    assert set(clean_df[config.TARGET].unique()) <= {0, 1}
    assert "num" not in clean_df.columns


def test_clean_keeps_expected_columns(clean_df):
    assert list(clean_df.columns) == config.FEATURES + [config.TARGET]


def test_question_marks_become_nan(clean_df):
    assert clean_df["ca"].isna().sum() == 4
    assert clean_df["thal"].isna().sum() == 2


def test_all_features_are_numeric(clean_df):
    assert all(pd.api.types.is_numeric_dtype(clean_df[c]) for c in config.FEATURES)


def test_invalid_values_set_to_nan(raw_df):
    raw_df.loc[10, "chol"] = 5000      # impossible cholesterol
    raw_df.loc[11, "cp"] = 9           # unknown chest-pain code
    out = clean(raw_df)
    assert np.isnan(out.loc[10, "chol"])
    assert np.isnan(out.loc[11, "cp"])


def test_duplicates_removed(raw_df):
    doubled = pd.concat([raw_df, raw_df.head(5)], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_df))


def test_parse_uci_format():
    text = "63.0,1.0,1.0,145.0,233.0,1.0,2.0,150.0,0.0,2.3,3.0,0.0,6.0,0\n" \
           "67.0,1.0,4.0,160.0,286.0,0.0,2.0,108.0,1.0,1.5,2.0,?,3.0,2\n"
    df = parse_uci(text)
    assert df.shape == (2, 14)
    assert np.isnan(df.loc[1, "ca"])


def test_parse_mirror_maps_text_codes():
    text = (",Age,Sex,ChestPain,RestBP,Chol,Fbs,RestECG,MaxHR,ExAng,Oldpeak,"
            "Slope,Ca,Thal,AHD\n"
            "1,63,1,typical,145,233,1,2,150,0,2.3,3,0,fixed,No\n"
            "2,67,1,asymptomatic,160,286,0,2,108,1,1.5,2,3,normal,Yes\n")
    df = parse_mirror(text)
    assert df["cp"].tolist() == [1, 4]
    assert df["thal"].tolist() == [6, 3]
    assert df["num"].tolist() == [0, 1]
