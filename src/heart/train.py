"""Train, tune and compare models with MLflow tracking.

Run:  python -m heart.train
- Splits data into train (80%) and a held-out test set (20%).
- Tunes each model with GridSearchCV (5-fold stratified CV, scored on ROC-AUC).
- Logs every grid candidate as a child run, and the tuned model as a parent run
  (params, CV + test metrics, plots, the model itself) in MLflow.
- Picks the winner by mean CV ROC-AUC (not test score, to avoid "peeking"),
  saves it to models/model.joblib and registers it in the MLflow registry.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,  # noqa: E402
                             accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,  # noqa: E402
                                     cross_validate, train_test_split)

from heart import config  # noqa: E402
from heart.data import load_clean  # noqa: E402
from heart.features import build_pipeline  # noqa: E402

log = logging.getLogger(__name__)
# Rare category values (e.g. restecg=1, only ~4 patients) may be missing from a
# CV fold; the encoder safely maps them to all-zeros, so we silence that warning.
warnings.filterwarnings("ignore", message="Found unknown categories")
# Types skops must accept when MLflow saves our own (self-trained) models.
TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree",
                 "sklearn._loss.loss.HalfBinomialLoss",
                 "sklearn._loss.link.LogitLink", "sklearn._loss.link.Interval"]
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

# Each candidate model and the settings we search over.
CANDIDATES = {
    "logistic_regression": (
        LogisticRegression(max_iter=2000, solver="liblinear",
                           random_state=config.RANDOM_STATE),
        {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]},
    ),
    "random_forest": (
        RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [200, 400], "model__max_depth": [3, 5, None],
         "model__min_samples_leaf": [1, 5]},
    ),
    "gradient_boosting": (
        GradientBoostingClassifier(random_state=config.RANDOM_STATE),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [2, 3]},
    ),
}


def split(df: pd.DataFrame):
    X, y = df[config.FEATURES], df[config.TARGET]
    return train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                            random_state=config.RANDOM_STATE)


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


def _plots(model, name, X_test, y_test, out_dir) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax,
                                          display_labels=["No", "Yes"],
                                          cmap="Blues")
    ax.set_title(f"Confusion matrix - {name}")
    paths.append(os.path.join(out_dir, f"confusion_matrix_{name}.png"))
    fig.tight_layout(); fig.savefig(paths[-1], dpi=120); plt.close(fig)  # noqa: E702

    fig, ax = plt.subplots(figsize=(4.5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(f"ROC curve - {name}")
    paths.append(os.path.join(out_dir, f"roc_curve_{name}.png"))
    fig.tight_layout(); fig.savefig(paths[-1], dpi=120); plt.close(fig)  # noqa: E702

    importance = feature_importance(model)
    if importance is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        importance.head(15).sort_values().plot.barh(ax=ax, color="#2E86AB")
        ax.set_title(f"Top features - {name}")
        paths.append(os.path.join(out_dir, f"feature_importance_{name}.png"))
        fig.tight_layout(); fig.savefig(paths[-1], dpi=120); plt.close(fig)  # noqa: E702
    return paths


def feature_importance(pipeline) -> pd.Series | None:
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    est = pipeline.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(est.coef_[0])
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def tune_and_log(name, estimator, grid, X_train, y_train, X_test, y_test) -> dict:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    search = GridSearchCV(build_pipeline(estimator), grid, cv=cv,
                          scoring="roc_auc", n_jobs=-1)

    with mlflow.start_run(run_name=name) as parent:
        search.fit(X_train, y_train)

        # 1) every grid candidate as a child run
        res = search.cv_results_
        for i, params in enumerate(res["params"]):
            with mlflow.start_run(run_name=f"{name}-cand-{i}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                mlflow.log_metrics({"cv_roc_auc_mean": res["mean_test_score"][i],
                                    "cv_roc_auc_std": res["std_test_score"][i]})

        # 2) tuned model: full CV metrics + held-out test metrics
        best = search.best_estimator_
        cv_scores = cross_validate(best, X_train, y_train, cv=cv, scoring=SCORING)
        cv_metrics = {f"cv_{m}_mean": float(np.mean(cv_scores[f"test_{m}"]))
                      for m in SCORING}
        cv_metrics.update({f"cv_{m}_std": float(np.std(cv_scores[f"test_{m}"]))
                           for m in SCORING})
        test_metrics = {f"test_{k}": float(v)
                        for k, v in evaluate(best, X_test, y_test).items()}

        mlflow.set_tags({"model_type": name, "stage": "tuned"})
        mlflow.log_params({"model_type": name, "cv_folds": config.CV_FOLDS,
                           "train_rows": len(X_train), "test_rows": len(X_test),
                           **{k.replace("model__", "best_"): v
                              for k, v in search.best_params_.items()}})
        mlflow.log_metrics({**cv_metrics, **test_metrics})

        plot_dir = os.path.join("reports", "figures", "models")
        for p in _plots(best, name, X_test, y_test, plot_dir):
            mlflow.log_artifact(p, artifact_path="plots")
        pd.DataFrame(res).to_csv(f"{plot_dir}/cv_results_{name}.csv", index=False)
        mlflow.log_artifact(f"{plot_dir}/cv_results_{name}.csv", artifact_path="cv")

        signature = infer_signature(X_train, best.predict_proba(X_train))
        model_info = mlflow.sklearn.log_model(
            best, name="model", signature=signature,
            input_example=X_train.head(3), skops_trusted_types=TRUSTED_TYPES)

        log.info("%-20s CV ROC-AUC %.3f | test ROC-AUC %.3f | test acc %.3f",
                 name, cv_metrics["cv_roc_auc_mean"], test_metrics["test_roc_auc"],
                 test_metrics["test_accuracy"])
        return {"name": name, "run_id": parent.info.run_id, "model": best,
                "model_uri": model_info.model_uri,
                "best_params": search.best_params_, **cv_metrics, **test_metrics}


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)

    results = [tune_and_log(n, est, grid, X_train, y_train, X_test, y_test)
               for n, (est, grid) in CANDIDATES.items()]

    comparison = pd.DataFrame([{k: v for k, v in r.items() if k != "model"}
                               for r in results])
    comparison = comparison.drop(columns=["model_uri"])
    comparison.to_csv("reports/model_comparison.csv", index=False)
    print(comparison[["name", "cv_roc_auc_mean", "cv_accuracy_mean",
                      "test_roc_auc", "test_accuracy", "test_recall"]]
          .round(3).to_string(index=False))

    winner = max(results, key=lambda r: r["cv_roc_auc_mean"])
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(winner["model"], config.MODEL_PATH)

    meta = {
        "model_name": winner["name"],
        "mlflow_run_id": winner["run_id"],
        "best_params": {k.replace("model__", ""): v
                        for k, v in winner["best_params"].items()},
        "metrics": {k: round(v, 4) for k, v in winner.items()
                    if k.startswith(("cv_", "test_"))},
        "features": config.FEATURES,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "model_version": os.getenv("MODEL_VERSION", "1.0.0"),
    }
    config.MODEL_META_PATH.write_text(json.dumps(meta, indent=2, default=str))

    try:
        mlflow.register_model(winner["model_uri"], "heart-disease-model")
    except Exception as exc:  # registry is optional (e.g. plain file store)
        log.warning("Model registry skipped: %s", exc)

    log.info("Winner: %s -> saved to %s", winner["name"], config.MODEL_PATH)
    return meta


if __name__ == "__main__":
    main()
