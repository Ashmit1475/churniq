"""Per-customer explanations.

Primary path: SHAP. If shap is not installed, falls back to a model-agnostic
contribution estimate so the pipeline never hard-fails on an optional dependency.

The output is a short human-readable reason string per customer, e.g.
"month-to-month contract; tenure 3 months; electronic check" - which is what
makes the risk score usable by a retention team rather than just a number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pipeline import feature_names

# Engineered features whose column name does not read as English on its own.
# Everything else is humanised generically by _humanize().
_LABELS = {
    "is_month_to_month": "Month-to-month contract",
    "has_auto_payment": "Automatic payment",
    "is_new_customer": "First-year customer",
    "avg_monthly_spend": "Average monthly spend",
    "services_count": "Services subscribed",
    "charge_ratio": "Bill vs lifetime average",
    "tenure_bucket": "Tenure band",
}


def _humanize(col: str) -> str:
    """'MonthlyCharges' -> 'Monthly charges'; 'services_count' -> 'Services count'."""
    if col in _LABELS:
        return _LABELS[col]
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", col).replace("_", " ").strip()
    return spaced[:1].upper() + spaced[1:] if spaced else col


def _split_categorical(name: str, cat_cols: Sequence[str]) -> tuple[str, str] | None:
    """Split 'tenure_bucket_0-12m' into ('tenure_bucket', '0-12m').

    Matches against the *actual* source column names, longest first. Partitioning
    on the first underscore instead would turn 'tenure_bucket_0-12m' into
    'tenure: bucket_0-12m' - the source column names are the only reliable guide
    because both column names and category values may contain underscores.
    """
    for col in sorted(cat_cols, key=len, reverse=True):
        if name.startswith(f"{col}_"):
            return col, name[len(col) + 1:]
    return None


def _source_columns(fitted_pipeline) -> dict[str, list[str]]:
    """Map each ColumnTransformer branch name to the source columns it consumed."""
    prep = fitted_pipeline.named_steps["prep"]
    out: dict[str, list[str]] = {}
    for name, _, columns in getattr(prep, "transformers_", []):
        if isinstance(columns, (list, tuple)):
            out[name] = list(columns)
    return out


def _prettify(raw_name: str, cat_cols: Sequence[str] = ()) -> str:
    """Turn 'cat__Contract_Month-to-month' into 'Contract: Month-to-month'.

    Numeric features keep their identity ('num__is_month_to_month' ->
    'Month-to-month contract'); only one-hot columns are split into
    'column: value'.
    """
    branch, sep, name = raw_name.partition("__")
    if not sep:
        branch, name = "", raw_name

    if branch == "cat":
        split = _split_categorical(name, cat_cols)
        if split:
            col, val = split
            return f"{_humanize(col)}: {val}"
    elif branch == "num":
        return _humanize(name)

    # Unknown branch: fall back to a conservative first-underscore split.
    col, _, val = name.partition("_")
    return f"{_humanize(col)}: {val}" if val else _humanize(name)


def pretty_feature_names(fitted_pipeline) -> list[str]:
    """Display-ready labels aligned 1:1 with feature_names(fitted_pipeline)."""
    cat_cols = _source_columns(fitted_pipeline).get("cat", [])
    return [_prettify(n, cat_cols) for n in feature_names(fitted_pipeline)]


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
    """For each row, the k features pushing risk *up* the most.

    `names` are expected to be display-ready already (see pretty_feature_names).
    """
    reasons: list[str] = []
    order = np.argsort(-contributions, axis=1)[:, :k]
    for row_i, cols in enumerate(order):
        picks = [names[c] for c in cols if contributions[row_i, c] > 0]
        reasons.append("; ".join(picks) if picks else "no dominant risk driver")
    return reasons


def explain(fitted_pipeline, X: pd.DataFrame, max_samples: int = 2000, k: int = 3):
    """Return (reason strings, method used)."""
    contrib, names = shap_contributions(fitted_pipeline, X, max_samples)
    method = "shap"
    if contrib is None:
        contrib, names = fallback_contributions(fitted_pipeline, X)
        method = "importance_proxy"
    labels = pretty_feature_names(fitted_pipeline)
    return top_reasons(np.asarray(contrib), labels, k), method


def global_importance(
    fitted_pipeline, X: pd.DataFrame, max_samples: int = 2000, top: int = 15
) -> pd.DataFrame:
    """Mean absolute contribution per feature - the 'top churn drivers' chart."""
    contrib, names = shap_contributions(fitted_pipeline, X, max_samples)
    if contrib is None:
        contrib, names = fallback_contributions(fitted_pipeline, X)
    mean_abs = np.abs(np.asarray(contrib)).mean(axis=0)
    return (
        pd.DataFrame(
            {"feature": pretty_feature_names(fitted_pipeline), "importance": mean_abs}
        )
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
