"""Stage 2: SQL JOIN -> engineered feature table.

Aggregation and joining happen in SQL; only derived per-row features are
computed in pandas. Each engineered feature has a stated reason - be ready to
justify all of them in an interview.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import Config, load_config
from db import get_engine, read_sql

# Columns produced by add_engineered_features()
ENGINEERED = [
    "tenure_bucket",
    "avg_monthly_spend",
    "services_count",
    "is_month_to_month",
    "has_auto_payment",
    "charge_ratio",
    "is_new_customer",
]


def join_query(cfg: Config) -> str:
    d = cfg["database"]
    return f"""
        SELECT c.*,
               b."PaperlessBilling",
               b."PaymentMethod",
               b."MonthlyCharges",
               b."TotalCharges"
        FROM {d['customers_table']} AS c
        INNER JOIN {d['billing_table']} AS b
                ON c."customerID" = b."customerID"
    """


def load_joined(cfg: Config) -> pd.DataFrame:
    engine = get_engine(cfg)
    return read_sql(join_query(cfg), engine)


def add_engineered_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Derive the features that carry analytical judgment.

    Why each one exists:
      tenure_bucket      - churn risk is non-linear in tenure; buckets capture the cliff
      avg_monthly_spend  - detects customers whose spend has drifted from their bill
      services_count     - bundling is strongly protective against churn
      is_month_to_month  - typically the single strongest churn signal
      has_auto_payment   - electronic-check payers churn far more than auto-pay
      charge_ratio       - current bill vs lifetime average -> recent price increase
      is_new_customer    - first-year customers behave differently from the rest
    """
    fcfg = cfg["features"]
    out = df.copy()

    bins = fcfg["tenure_buckets"]["bins"]
    labels = fcfg["tenure_buckets"]["labels"]
    out["tenure_bucket"] = pd.cut(out["tenure"], bins=bins, labels=labels).astype(str)

    # Guard against tenure == 0 producing inf / NaN.
    safe_tenure = out["tenure"].replace(0, np.nan)
    out["avg_monthly_spend"] = (out["TotalCharges"] / safe_tenure).fillna(
        out["MonthlyCharges"]
    )

    service_cols = [c for c in fcfg["service_columns"] if c in out.columns]
    out["services_count"] = sum((out[c] == "Yes").astype(int) for c in service_cols)

    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["has_auto_payment"] = (
        out["PaymentMethod"].isin(fcfg["auto_payment_methods"]).astype(int)
    )

    denom = out["avg_monthly_spend"].replace(0, np.nan)
    out["charge_ratio"] = (out["MonthlyCharges"] / denom).fillna(1.0)
    out["charge_ratio"] = out["charge_ratio"].clip(0, 5)

    out["is_new_customer"] = (out["tenure"] <= 12).astype(int)

    return out


def split_columns(df: pd.DataFrame, cfg: Config) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) for the model, excluding id/target."""
    drop = {cfg["data"]["id_column"], cfg["data"]["target_column"]}
    feature_cols = [c for c in df.columns if c not in drop]
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def build_feature_table(cfg: Config) -> pd.DataFrame:
    df = load_joined(cfg)
    return add_engineered_features(df, cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the feature table")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = build_feature_table(cfg)

    out = cfg.path("features_parquet")
    # Parquet, not CSV: smaller, far faster to read, and preserves dtypes.
    try:
        df.to_parquet(out, index=False)
    except ImportError:
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
        print("  ! pyarrow not installed - wrote CSV instead (pip install pyarrow)")

    num, cat = split_columns(df, cfg)
    print(f"Feature table: {len(df):,} rows x {len(df.columns)} columns -> {out}")
    print(f"  {len(num)} numeric, {len(cat)} categorical")
    print(f"  engineered: {', '.join(ENGINEERED)}")


if __name__ == "__main__":
    main()
