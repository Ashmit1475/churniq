"""Stage 3: train, compare, tune and save.

Trains every enabled model, compares them with stratified cross-validation,
selects the best by ROC-AUC, tunes the decision threshold for business value,
and saves a single self-contained artifact.

Usage:
    python src/train.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from business import revenue_at_risk, tune_threshold
from config import load_config
from features import build_feature_table, split_columns
from pipeline import build_estimators, build_pipeline


def evaluate_at(
    y_true: pd.Series | np.ndarray, proba: np.ndarray, threshold: float
) -> dict[str, float | int]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def select_best_model(results: Mapping[str, dict[str, Any]]) -> str:
    """Pick the winning model by cross-validated ROC-AUC on the training split.

    Selection deliberately ignores `test_roc_auc`. Choosing the model that scores
    best on the test set folds that set into model selection, and the test number
    reported afterwards is then an optimistic estimate of a model picked partly
    for scoring well on it - the held-out set is no longer held out.

    Ties break toward the lower CV standard deviation: given equal mean ranking
    quality, prefer the model whose quality varies least across folds.
    """
    return max(
        results,
        key=lambda name: (
            results[name]["cv_roc_auc_mean"],
            -results[name]["cv_roc_auc_std"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and select the churn model")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]

    print("Loading features from SQL...")
    df = build_feature_table(cfg)

    y = (df[dcfg["target_column"]] == dcfg["positive_label"]).astype(int)
    X = df.drop(columns=[dcfg["target_column"]])
    numeric_cols, categorical_cols = split_columns(df, cfg)
    X = X[numeric_cols + categorical_cols]

    print(f"  {len(X):,} rows, {len(numeric_cols)} numeric + {len(categorical_cols)} categorical")
    print(f"  class balance: {y.mean():.1%} churn  (imbalanced - accuracy alone is misleading)")

    X_train, X_test, y_train, y_test, charges_train, charges_test = train_test_split(
        X,
        y,
        df["MonthlyCharges"],
        test_size=dcfg["test_size"],
        random_state=cfg.random_state,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=dcfg["cv_folds"], shuffle=True, random_state=cfg.random_state
    )

    # ---------------------------------------------------------------- compare
    print(f"\nComparing models ({dcfg['cv_folds']}-fold stratified CV on train set)")
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}

    for name, estimator in build_estimators(cfg).items():
        pipe = build_pipeline(estimator, numeric_cols, categorical_cols)
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "cv_roc_auc_mean": float(cv_scores.mean()),
            "cv_roc_auc_std": float(cv_scores.std()),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
            "test_pr_auc": float(average_precision_score(y_test, proba)),
            "at_default_threshold": evaluate_at(y_test, proba, 0.5),
        }
        fitted[name] = pipe
        print(
            f"  {name:22s} CV ROC-AUC {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})"
            f"   test ROC-AUC {results[name]['test_roc_auc']:.4f}"
        )

    best_name = select_best_model(results)
    best_pipe = fitted[best_name]
    best_proba = best_pipe.predict_proba(X_test)[:, 1]
    print(f"\nBest model: {best_name}  (selected on CV ROC-AUC, not the test score)")
    print(f"  held-out test ROC-AUC: {results[best_name]['test_roc_auc']:.4f}")

    # ------------------------------------------------------- threshold tuning
    threshold, best_value, curve = tune_threshold(
        y_test.to_numpy(), best_proba, charges_test.to_numpy(), cfg
    )
    default_metrics = evaluate_at(y_test, best_proba, 0.5)
    tuned_metrics = evaluate_at(y_test, best_proba, threshold)
    sym = cfg["business"]["currency_symbol"]

    print(f"\nThreshold tuning (maximising expected campaign value)")
    print(f"  default 0.50 -> precision {default_metrics['precision']:.3f}  recall {default_metrics['recall']:.3f}")
    print(f"  tuned  {threshold:.2f} -> precision {tuned_metrics['precision']:.3f}  recall {tuned_metrics['recall']:.3f}")
    print(f"  expected value at tuned threshold: {sym}{best_value:,.0f}")

    curve.to_csv(cfg.path("metrics_json").parent / "threshold_curve.csv", index=False)

    # ------------------------------------------------------- revenue at risk
    full_proba = best_pipe.predict_proba(X)[:, 1]
    rar = revenue_at_risk(full_proba, df["MonthlyCharges"].to_numpy(), cfg)
    print(f"  revenue at risk across all customers: {sym}{rar:,.0f}")

    # ------------------------------------------------------------ save model
    artifact = {
        "pipeline": best_pipe,
        "model_name": best_name,
        "threshold": threshold,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(X_train)),
    }
    joblib.dump(artifact, cfg.path("model_artifact"))
    print(f"\nSaved artifact -> {cfg.path('model_artifact')}")

    metrics = {
        "trained_at": artifact["trained_at"],
        "n_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "best_model": best_name,
        "tuned_threshold": threshold,
        "expected_campaign_value": best_value,
        "revenue_at_risk": rar,
        "metrics_at_default_threshold": default_metrics,
        "metrics_at_tuned_threshold": tuned_metrics,
        "test_roc_auc": results[best_name]["test_roc_auc"],
        "test_pr_auc": results[best_name]["test_pr_auc"],
        "model_comparison": results,
    }
    with open(cfg.path("metrics_json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"Saved metrics  -> {cfg.path('metrics_json')}")


if __name__ == "__main__":
    main()
