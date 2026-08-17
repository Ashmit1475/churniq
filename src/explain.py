"""Per-customer explanations.

Primary path: SHAP. If shap is not installed, falls back to a model-agnostic
contribution estimate so the pipeline never hard-fails on an optional dependency.

The output is a short human-readable reason string per customer, e.g.
"month-to-month contract; tenure 3 months; electronic check" - which is what
makes the risk score usable by a retention team rather than just a number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import feature_names


def _prettify(raw_name: str) -> str:
    """Turn 'cat__Contract_Month-to-month' into 'Contract: Month-to-month'."""
    name = raw_name.split("__", 1)[-1]
    if "_" in name:
        col, _, val = name.partition("_")
        if val:
            return f"{col}: {val}"
    return name


def shap_contributions(fitted_pipeline, X: pd.DataFrame, max_samples: int = 2000):
    """Return (matrix of per-feature contributions, feature names) or (None, names).

    Explainer choice matters for runtime:
      * Tree models  -> TreeExplainer in tree_path_dependent mode (no background
        dataset). This is near-instant. Passing a 2,000-row background instead
        would make SHAP roughly O(background x samples) and turn a two-second
        step into several minutes.
      * Linear models -> LinearExplainer with a small background sample.
    """
    names = feature_names(fitted_pipeline)
    try:
        import shap
    except ImportError:
        return None, names

    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    Xt = np.asarray(prep.transform(X))

    try:
        if hasattr(clf, "feature_importances_") or hasattr(clf, "estimators_"):
            explainer = shap.TreeExplainer(clf)
        elif hasattr(clf, "coef_"):
            # A small background is enough for a linear model and keeps it fast.
            n_bg = min(200, len(Xt))
            idx = np.random.default_rng(0).choice(len(Xt), n_bg, replace=False)
            explainer = shap.LinearExplainer(clf, Xt[idx])
        else:
            n_bg = min(100, len(Xt))
            idx = np.random.default_rng(0).choice(len(Xt), n_bg, replace=False)
            explainer = shap.Explainer(clf, Xt[idx])

        arr = np.asarray(explainer.shap_values(Xt))

        # Shapes vary by model and shap version:
        #   (n, features)                -> already what we want
        #   (n, features, 2) or (2, n, f) -> binary classifier, take positive class
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[-1] == 2 else arr[1]
        if arr.shape != Xt.shape:
            raise ValueError(f"unexpected SHAP shape {arr.shape} for input {Xt.shape}")
        return arr, names
    except Exception as exc:  # pragma: no cover - explainer support varies by model
        print(f"  ! SHAP unavailable for this model ({exc}); using importance proxy")
        return None, names


def fallback_contributions(fitted_pipeline, X: pd.DataFrame):
    """Approximate contributions when SHAP is unavailable.

    Uses (standardised feature value) x (global importance) as a rough per-row
    signal. Less rigorous than SHAP - clearly labelled as a fallback.
    """
    names = feature_names(fitted_pipeline)
    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    Xt = np.asarray(prep.transform(X), dtype=float)

    if hasattr(clf, "coef_"):
        weights = np.ravel(clf.coef_)
    elif hasattr(clf, "feature_importances_"):
        weights = np.asarray(clf.feature_importances_, dtype=float)
    else:
        weights = np.ones(Xt.shape[1])

    if weights.shape[0] != Xt.shape[1]:
        weights = np.ones(Xt.shape[1])

    return Xt * weights, names


def top_reasons(
    contributions: np.ndarray, names: list[str], k: int = 3
) -> list[str]:
    """For each row, the k features pushing risk *up* the most."""
    reasons: list[str] = []
    order = np.argsort(-contributions, axis=1)[:, :k]
    for row_i, cols in enumerate(order):
        picks = [
            _prettify(names[c]) for c in cols if contributions[row_i, c] > 0
        ]
        reasons.append("; ".join(picks) if picks else "no dominant risk driver")
    return reasons


def explain(fitted_pipeline, X: pd.DataFrame, max_samples: int = 2000, k: int = 3):
    """Return (reason strings, method used)."""
    contrib, names = shap_contributions(fitted_pipeline, X, max_samples)
    method = "shap"
    if contrib is None:
        contrib, names = fallback_contributions(fitted_pipeline, X)
        method = "importance_proxy"
    return top_reasons(np.asarray(contrib), names, k), method


def global_importance(
    fitted_pipeline, X: pd.DataFrame, max_samples: int = 2000, top: int = 15
) -> pd.DataFrame:
    """Mean absolute contribution per feature - the 'top churn drivers' chart."""
    contrib, names = shap_contributions(fitted_pipeline, X, max_samples)
    if contrib is None:
        contrib, names = fallback_contributions(fitted_pipeline, X)
    mean_abs = np.abs(np.asarray(contrib)).mean(axis=0)
    return (
        pd.DataFrame({"feature": [_prettify(n) for n in names], "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
