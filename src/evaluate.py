"""Stage 4: evaluation figures for the README and the dashboard.

Produces: ROC curve, precision-recall curve, confusion matrix, threshold-value
curve, and the top churn drivers chart.

Usage:
    python src/evaluate.py
"""

from __future__ import annotations

import argparse
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless - required for CI and for WSL/servers
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from business import tune_threshold
from config import load_config
from explain import global_importance
from features import build_feature_table, split_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation figures")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    figdir = cfg.path("figures_dir")
    figdir.mkdir(parents=True, exist_ok=True)

    artifact = joblib.load(cfg.path("model_artifact"))
    pipe = artifact["pipeline"]
    threshold = artifact["threshold"]

    df = build_feature_table(cfg)
    y = (df[dcfg["target_column"]] == dcfg["positive_label"]).astype(int)
    numeric_cols, categorical_cols = split_columns(df, cfg)
    X = df[numeric_cols + categorical_cols]

    # Same split as training so the figures describe genuinely held-out data.
    X_train, X_test, y_train, y_test, _, charges_test = train_test_split(
        X, y, df["MonthlyCharges"],
        test_size=dcfg["test_size"], random_state=cfg.random_state, stratify=y,
    )
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    # ---- ROC ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=ax, name=artifact["model_name"])
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="random")
    ax.set_title("ROC curve (held-out test set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "roc_curve.png", dpi=140)
    plt.close(fig)

    # ---- Precision-Recall ----
    fig, ax = plt.subplots(figsize=(5, 4.5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax, name=artifact["model_name"])
    ax.set_title("Precision-Recall curve")
    fig.tight_layout()
    fig.savefig(figdir / "pr_curve.png", dpi=140)
    plt.close(fig)

    # ---- Confusion matrix at the tuned threshold ----
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["Stayed", "Churned"], ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix (threshold = {threshold:.2f})")
    fig.tight_layout()
    fig.savefig(figdir / "confusion_matrix.png", dpi=140)
    plt.close(fig)

    # ---- Threshold vs expected value ----
    _, _, curve = tune_threshold(y_test.to_numpy(), proba, charges_test.to_numpy(), cfg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(curve["threshold"], curve["expected_value"], linewidth=2)
    ax.axvline(threshold, color="crimson", linestyle="--", label=f"chosen = {threshold:.2f}")
    ax.axvline(0.5, color="grey", linestyle=":", label="default = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Expected campaign value ({cfg['business']['currency_symbol'].strip()})")
    ax.set_title("Business value by decision threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "threshold_value.png", dpi=140)
    plt.close(fig)

    # ---- Top churn drivers ----
    imp = global_importance(pipe, X_test, cfg["explainability"]["max_samples"], top=15)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1])
    ax.set_xlabel("Mean absolute contribution")
    ax.set_title("Top 15 churn drivers")
    fig.tight_layout()
    fig.savefig(figdir / "churn_drivers.png", dpi=140)
    plt.close(fig)
    imp.to_csv(cfg.path("metrics_json").parent / "churn_drivers.csv", index=False)

    # ---- Model comparison chart from saved metrics ----
    metrics_path = cfg.path("metrics_json")
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as fh:
            metrics = json.load(fh)
        comp = metrics.get("model_comparison", {})
        if comp:
            names = list(comp)
            means = [comp[n]["cv_roc_auc_mean"] for n in names]
            errs = [comp[n]["cv_roc_auc_std"] for n in names]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(names, means, yerr=errs, capsize=5)
            ax.set_ylim(min(means) - 0.05, 1.0)
            ax.set_ylabel("CV ROC-AUC")
            ax.set_title("Model comparison (5-fold stratified CV)")
            plt.xticks(rotation=15, ha="right")
            fig.tight_layout()
            fig.savefig(figdir / "model_comparison.png", dpi=140)
            plt.close(fig)

    print(f"Figures written to {figdir}")
    for f in sorted(figdir.glob("*.png")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
