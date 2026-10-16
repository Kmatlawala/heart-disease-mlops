"""Exploratory Data Analysis: saves charts to reports/figures/.

Run:  python -m heart.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean  # noqa: E402

sns.set_theme(style="whitegrid", palette="Set2")
LABELS = {0: "No disease", 1: "Disease"}


def _save(fig, name: str) -> str:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def class_balance(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[config.TARGET].map(LABELS).value_counts()
    ax.bar(counts.index, counts.values, color=["#4C9F70", "#D1495B"])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    return _save(fig, "01_class_balance.png")


def numeric_histograms(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=df[config.TARGET].map(LABELS),
                     kde=True, ax=ax, element="step", stat="density",
                     common_norm=False)
        ax.set_title(col)
    fig.suptitle("Numeric features by outcome", fontsize=14)
    return _save(fig, "02_numeric_histograms.png")


def categorical_rates(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), config.CATEGORICAL_FEATURES):
        rate = df.groupby(col)[config.TARGET].mean().sort_index()
        ax.bar(rate.index.astype(int).astype(str), rate.values, color="#D1495B")
        ax.set_ylim(0, 1)
        ax.set_title(f"{col}: disease rate")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Share of patients with disease, per category", fontsize=14)
    return _save(fig, "03_categorical_rates.png")


def correlation_heatmap(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap")
    return _save(fig, "04_correlation_heatmap.png")


def boxplots(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 6, figsize=(16, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=df, x=df[config.TARGET].map(LABELS), y=col, ax=ax)
        ax.set_xlabel("")
    fig.suptitle("Outliers and spread by outcome", fontsize=14)
    return _save(fig, "05_boxplots.png")


def missing_values(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    miss = df.isna().sum()
    ax.bar(miss.index, miss.values, color="#EDAE49")
    ax.set_title("Missing values per column (filled inside the pipeline)")
    ax.tick_params(axis="x", rotation=45)
    return _save(fig, "00_missing_values.png")


def run_all() -> list[str]:
    df = load_clean()
    paths = [missing_values(df), class_balance(df), numeric_histograms(df),
             categorical_rates(df), correlation_heatmap(df), boxplots(df)]
    summary = df.describe().T.round(2)
    summary.to_csv(config.FIGURES_DIR / "summary_stats.csv")
    return paths


if __name__ == "__main__":
    for p in run_all():
        print("saved", p)
