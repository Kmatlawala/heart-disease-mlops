"""Tests for the preprocessing pipeline, training helpers and prediction code."""
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from heart import config
from heart.features import build_pipeline, build_preprocessor
from heart.predict import SAMPLE, predict_records
from heart.train import evaluate, feature_importance, split


def test_preprocessor_fills_missing_and_scales(clean_df):
    out = build_preprocessor().fit_transform(clean_df[config.FEATURES])
    assert not np.isnan(out).any()
    # first columns are the scaled numeric features -> mean ~0
    assert np.allclose(out[:, :len(config.NUMERIC_FEATURES)].mean(axis=0), 0, atol=1e-6)


def test_unknown_category_does_not_crash(trained_pipeline):
    bad = dict(SAMPLE, restecg=9)
    assert len(predict_records([bad], model=trained_pipeline)) == 1


def test_split_is_stratified(clean_df):
    X_train, X_test, y_train, y_test = split(clean_df)
    assert len(X_test) == round(len(clean_df) * config.TEST_SIZE)
    assert abs(y_train.mean() - y_test.mean()) < 0.1


def test_evaluate_returns_all_metrics(trained_pipeline, clean_df):
    m = evaluate(trained_pipeline, clean_df[config.FEATURES], clean_df[config.TARGET])
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= v <= 1 for v in m.values())
    assert m["roc_auc"] > 0.6  # learns the synthetic signal


def test_feature_importance_for_two_model_types(clean_df, trained_pipeline):
    rf = build_pipeline(RandomForestClassifier(n_estimators=20, random_state=0))
    rf.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    for pipe in (trained_pipeline, rf):
        imp = feature_importance(pipe)
        assert imp is not None and len(imp) > len(config.FEATURES)


def test_predict_output_format(trained_pipeline):
    out = predict_records([SAMPLE, SAMPLE], model=trained_pipeline)[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1
    assert 0 <= out["probability_disease"] <= 1


def test_model_save_and_reload_gives_same_result(trained_pipeline, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(trained_pipeline, path)
    again = joblib.load(path)
    assert predict_records([SAMPLE], model=again) == \
        predict_records([SAMPLE], model=trained_pipeline)
