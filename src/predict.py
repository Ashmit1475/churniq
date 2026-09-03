"""Stage 5: score customers and write results back to SQL.

This is the inference half of the train/inference split. It never retrains -
it loads the saved artifact and applies it. That is what makes the model
reusable on new data rather than tied to the rows it was trained on.

Score the customers already in the database:
    python src/predict.py

Score a brand-new CSV with the same schema:
    python src/predict.py --csv data/raw/new_customers.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from business import customer_value, revenue_at_risk, simulate_campaign
from config import Config, load_config
from db import create_indexes, get_engine, write_table
from explain import explain
from features import add_engineered_features, build_feature_table
from ingest import clean


def score(df_features: pd.DataFrame, artifact: dict, cfg: Config) -> pd.DataFrame:
    """Apply the saved pipeline and assemble the output table."""
    pipe = artifact["pipeline"]
    threshold = artifact["threshold"]
    cols = artifact["numeric_cols"] + artifact["categorical_cols"]

    missing = [c for c in cols if c not in df_features.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} column(s) the model expects: {missing[:8]}"
        )

    X = df_features[cols]
    proba = pipe.predict_proba(X)[:, 1]

    out = pd.DataFrame(
        {
            cfg["data"]["id_column"]: df_features[cfg["data"]["id_column"]].to_numpy(),
            "churn_probability": np.round(proba, 4),
            "risk_flag": np.where(proba >= threshold, "High", "Low"),
            "risk_band": pd.cut(
                proba,
                bins=[-0.001, 0.25, 0.50, 0.75, 1.0],
                labels=["Low", "Medium", "High", "Critical"],
            ).astype(str),
            "threshold_used": threshold,
            "model_name": artifact["model_name"],
        }
    )

    # Carry through the business columns the dashboard needs.
    for col in ["MonthlyCharges", "tenure", "Contract", "PaymentMethod",
                "InternetService", "tenure_bucket", "services_count"]:
        if col in df_features.columns:
            out[col] = df_features[col].to_numpy()

    if "MonthlyCharges" in out.columns:
        out["customer_value"] = np.round(
            customer_value(out["MonthlyCharges"].to_numpy(), cfg), 2
        )
        out["revenue_at_risk"] = np.round(
            out["churn_probability"] * out["customer_value"], 2
        )

    if cfg["explainability"]["enabled"]:
        reasons, method = explain(pipe, X, k=cfg["explainability"]["top_reasons"])
        out["top_risk_reasons"] = reasons
        out["explanation_method"] = method

    if cfg["data"]["target_column"] in df_features.columns:
        out["actual_churn"] = df_features[cfg["data"]["target_column"]].to_numpy()

    return out.sort_values("churn_probability", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score customers for churn risk")
    parser.add_argument("--config", default=None)
    parser.add_argument("--csv", default=None, help="score a new CSV instead of the DB")
    parser.add_argument("--top", type=int, default=500, help="campaign simulation size")
    args = parser.parse_args()

    cfg = load_config(args.config)
    artifact = joblib.load(cfg.path("model_artifact"))
    print(f"Loaded {artifact['model_name']} (trained {artifact['trained_at'][:19]}Z)")
    print(f"  decision threshold: {artifact['threshold']:.2f}")

    if args.csv:
        path = Path(args.csv)
        print(f"Scoring new file: {path}")
        raw = clean(pd.read_csv(path))
        features = add_engineered_features(raw, cfg)
    else:
        print("Scoring customers from the database")
        features = build_feature_table(cfg)

    scored = score(features, artifact, cfg)

    engine = get_engine(cfg)
    table = cfg["database"]["predictions_table"]
    write_table(scored, table, engine)
    create_indexes(engine, [(table, cfg["data"]["id_column"]), (table, "risk_flag")])
    scored.to_csv(cfg.path("scored_csv"), index=False)

    sym = cfg["business"]["currency_symbol"]
    rar = revenue_at_risk(
        scored["churn_probability"].to_numpy(), scored["MonthlyCharges"].to_numpy(), cfg
    )
    n_high = int((scored["risk_flag"] == "High").sum())

    print(f"\nScored {len(scored):,} customers -> table '{table}' and {cfg.path('scored_csv')}")
    print(f"  flagged high risk: {n_high:,} ({n_high / len(scored):.1%})")
    print(f"  total revenue at risk: {sym}{rar:,.0f}")
    print("\n  risk band distribution:")
    for band, count in scored["risk_band"].value_counts().items():
        print(f"    {band:9s} {count:5,d}")

    sim = simulate_campaign(scored, args.top, cfg)
    print(f"\nRetention campaign simulation (top {sim['customers_targeted']} riskiest):")
    print(f"  campaign cost          {sym}{sim['campaign_cost']:,.0f}")
    print(f"  expected revenue saved {sym}{sim['expected_revenue_saved']:,.0f}")
    print(f"  net benefit            {sym}{sim['net_benefit']:,.0f}   (ROI {sim['roi']:.1%})")


if __name__ == "__main__":
    main()
