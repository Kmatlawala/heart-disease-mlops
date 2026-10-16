"""Draws the architecture diagram used in the README and report.

Run:  python scripts/architecture_diagram.py  -> reports/figures/architecture.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

COLORS = {"data": "#E8F1FA", "train": "#EAF6EC", "ci": "#FFF4E0",
          "serve": "#F3ECFA", "mon": "#FDECEC"}
EDGE = {"data": "#2E86AB", "train": "#3C8D4F", "ci": "#D08C00",
        "serve": "#7B4FB0", "mon": "#C0392B"}


def box(ax, x, y, w, h, title, lines, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=COLORS[kind], ec=EDGE[kind], lw=1.8))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top",
            fontsize=10.5, weight="bold", color=EDGE[kind])
    ax.text(x + w / 2, y + h - 0.68, "\n".join(lines), ha="center", va="top",
            fontsize=8.6, color="#222", linespacing=1.45)


def arrow(ax, a, b, label="", rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color="#555", connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.15, label, ha="center",
                fontsize=8, color="#555", style="italic")


fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.2)
ax.axis("off")
ax.text(7.5, 7.95, "Heart Disease Risk API - MLOps Architecture", ha="center",
        fontsize=15, weight="bold")

# Row 1: data -> training -> artefacts
box(ax, 0.3, 5.0, 2.9, 2.4, "1. Data", ["UCI Heart Disease", "download script (+ mirror)",
                                         "clean.py: types, ranges,", "duplicates, binary target"], "data")
box(ax, 3.7, 5.0, 3.2, 2.4, "2. EDA + Training", ["EDA charts (heart.eda)", "sklearn Pipeline:",
                                                   "impute -> scale / one-hot", "LR, RF, GB + GridSearchCV",
                                                   "5-fold stratified CV"], "train")
box(ax, 7.4, 5.0, 3.0, 2.4, "3. MLflow", ["params, metrics, plots", "child run per grid point",
                                           "model + signature", "model registry"], "train")
box(ax, 10.9, 5.0, 3.8, 2.4, "4. Model artefact", ["models/model.joblib", "(preprocessing + model)",
                                                     "model_meta.json", "pinned requirements"], "train")

# Row 2: CI/CD
box(ax, 0.3, 2.35, 6.6, 2.1, "5. GitHub Actions CI/CD", [
    "push / PR  ->  lint (flake8)  ->  pytest + coverage  ->  train + quality gate",
    "->  docker build + smoke test /health /predict  ->  push to GHCR",
    "fails on any lint / test / quality error; uploads logs + artefacts"], "ci")
box(ax, 7.4, 2.35, 3.0, 2.1, "6. Docker image", ["python:3.12-slim, non-root", "FastAPI + uvicorn",
                                                  "HEALTHCHECK /health"], "serve")
box(ax, 10.9, 2.35, 3.8, 2.1, "7. Kubernetes", ["Deployment (2 replicas,", "probes, rolling update)",
                                                 "Service: LoadBalancer", "Ingress, HPA (optional)"], "serve")

# Row 3: users + monitoring
box(ax, 0.3, 0.2, 4.2, 1.75, "Client", ["POST /predict  (JSON)", "-> prediction, confidence"], "data")
box(ax, 10.4, 0.2, 4.3, 1.75, "8. Monitoring", ["/metrics -> Prometheus -> Grafana",
                                               "JSON request logs (kubectl logs)"], "mon")
box(ax, 5.2, 0.2, 4.6, 1.75, "API endpoints", ["/predict  /predict/batch  /health",
                                                "/model-info  /metrics  /docs"], "serve")

arrow(ax, (3.2, 6.2), (3.7, 6.2))
arrow(ax, (6.9, 6.2), (7.4, 6.2), "log")
arrow(ax, (10.4, 6.2), (10.9, 6.2))
arrow(ax, (12.8, 5.0), (12.8, 4.45), "")
arrow(ax, (12.0, 5.0), (8.9, 4.45))
ax.text(9.3, 4.75, "copied into image", fontsize=8, color="#555", style="italic")
arrow(ax, (6.9, 3.4), (7.4, 3.4), "builds")
arrow(ax, (10.4, 3.4), (10.9, 3.4), "deploy")
arrow(ax, (4.5, 1.1), (5.2, 1.1))
arrow(ax, (9.8, 1.1), (10.4, 1.1))
ax.text(10.1, 1.3, "scrape", ha="center", fontsize=8, color="#555", style="italic")
arrow(ax, (11.6, 2.35), (8.8, 1.95), "", rad=0.0)
ax.text(10.5, 2.25, "serves", fontsize=8, color="#555", style="italic")

fig.savefig("reports/figures/architecture.png", dpi=150, bbox_inches="tight")
print("saved reports/figures/architecture.png")
