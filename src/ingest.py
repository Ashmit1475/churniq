"""Stage 1: raw CSV -> normalised SQL tables.

Splits the flat Telco CSV into two tables so the pipeline reads from a database
via a JOIN rather than from a CSV in pandas:

    customers : demographics, tenure, contract, services
    billing   : charges, payment method, paperless billing

This split is deliberately introduced (the source file is flat) - it is here to
demonstrate schema design, joins and indexing. Documented honestly in the README.

Usage:
    python src/ingest.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from config import load_config
from db import create_indexes, get_engine, write_table

CUSTOMER_COLS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "Churn",
]
BILLING_COLS = [
    "customerID", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges",
]


def _to_int(df: pd.DataFrame, col: str) -> pd.Series:
    """Integer conversion that says which column failed, and how badly.

    A plain `.astype("int64")` on a column holding one unparseable value raises
    pandas' "Cannot convert non-finite values (NA or inf) to integer" - which
    names neither the column nor the rows. On a real extract with a handful of
    blank tenures that is a dead end. Refusing loudly is the right call here:
    silently filling a tenure or a senior-citizen flag would change what the
    model is trained on without anyone noticing.
    """
    values = pd.to_numeric(df[col], errors="coerce")
    n_bad = int(values.isna().sum())
    if n_bad:
        examples = df.loc[values.isna(), col].astype(str).unique()[:3].tolist()
        raise ValueError(
            f"{col}: {n_bad} of {len(df):,} rows are not numeric (e.g. {examples}). "
            f"Fix them in the source CSV - filling a value here would quietly "
            f"change the model's inputs."
        )
    return values.astype("int64")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix the known data-quality issues in the Telco dataset."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # TotalCharges arrives as text with blank strings where tenure == 0.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].astype(str).str.strip(), errors="coerce"
    )
    n_blank = int(df["TotalCharges"].isna().sum())
    if n_blank:
        # A customer with zero tenure has genuinely been charged nothing yet.
        df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
        print(f"  cleaned {n_blank} blank TotalCharges values (tenure = 0) -> 0.0")

    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["tenure"] = _to_int(df, "tenure")
    df["SeniorCitizen"] = _to_int(df, "SeniorCitizen")

    before = len(df)
    df = df.drop_duplicates(subset=["customerID"])
    if len(df) < before:
        print(f"  dropped {before - len(df)} duplicate customerID rows")

    # Trim stray whitespace in every categorical column.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest raw CSV into SQL")
    parser.add_argument("--config", default=None)
    parser.add_argument("--csv", default=None, help="override paths.raw_csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    csv_path = Path(args.csv) if args.csv else cfg.path("raw_csv")

    if not csv_path.exists():
        sys.exit(
            f"No CSV at {csv_path}\n"
            "Either place the Kaggle Telco-Customer-Churn.csv there, or run:\n"
            "    python src/make_sample_data.py"
        )

    print(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows x {len(df.columns)} columns")

    missing = [c for c in CUSTOMER_COLS + BILLING_COLS if c not in df.columns]
    if missing:
        sys.exit(f"CSV is missing expected columns: {missing}")

    df = clean(df)

    engine = get_engine(cfg)
    dbcfg = cfg["database"]

    write_table(df[CUSTOMER_COLS], dbcfg["customers_table"], engine)
    write_table(df[BILLING_COLS], dbcfg["billing_table"], engine)
    create_indexes(
        engine,
        [
            (dbcfg["customers_table"], "customerID"),
            (dbcfg["customers_table"], "Contract"),
            (dbcfg["billing_table"], "customerID"),
        ],
    )

    print(f"Wrote tables '{dbcfg['customers_table']}' and '{dbcfg['billing_table']}'")
    print(f"  -> {cfg.db_url}")
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"  churn rate: {churn_rate:.1%}")


if __name__ == "__main__":
    main()
