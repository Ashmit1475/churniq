"""Generate a schema-identical synthetic Telco churn dataset.

Why this exists
---------------
The real dataset (IBM Telco Customer Churn) requires a Kaggle login, so the repo
would not run out of the box for anyone who clones it. This generator produces a
CSV with the *exact same 21 columns and category values*, with churn driven by
realistic relationships (month-to-month contracts, short tenure and electronic
check payment all raise churn probability).

Swap in the real file at any time - `ingest.py` reads whatever CSV is at
`paths.raw_csv` and does not care which one it is.

Usage:
    python src/make_sample_data.py --rows 7043
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from config import load_config

YES_NO = ["Yes", "No"]
INTERNET = ["DSL", "Fiber optic", "No"]
CONTRACT = ["Month-to-month", "One year", "Two year"]
PAYMENT = [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)",
]


def _dependent_service(rng, internet, n):
    """Service add-ons are 'No internet service' when the customer has no internet."""
    out = rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]).astype(object)
    out[internet == "No"] = "No internet service"
    return out


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, size=n)
    contract = rng.choice(CONTRACT, size=n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(INTERNET, size=n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(PAYMENT, size=n, p=[0.34, 0.23, 0.22, 0.21])
    paperless = rng.choice(YES_NO, size=n, p=[0.59, 0.41])
    phone = rng.choice(YES_NO, size=n, p=[0.90, 0.10])

    multiple = rng.choice(["Yes", "No"], size=n, p=[0.42, 0.58]).astype(object)
    multiple[phone == "No"] = "No phone service"

    df = pd.DataFrame(
        {
            "customerID": [f"{i:04d}-CHRNQ" for i in range(1, n + 1)],
            "gender": rng.choice(["Male", "Female"], size=n),
            "SeniorCitizen": rng.choice([0, 1], size=n, p=[0.84, 0.16]),
            "Partner": rng.choice(YES_NO, size=n, p=[0.48, 0.52]),
            "Dependents": rng.choice(YES_NO, size=n, p=[0.30, 0.70]),
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": _dependent_service(rng, internet, n),
            "OnlineBackup": _dependent_service(rng, internet, n),
            "DeviceProtection": _dependent_service(rng, internet, n),
            "TechSupport": _dependent_service(rng, internet, n),
            "StreamingTV": _dependent_service(rng, internet, n),
            "StreamingMovies": _dependent_service(rng, internet, n),
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
        }
    )

    # ---- Monthly charges: base fee by internet tier plus per-service add-ons ----
    base = np.select(
        [internet == "Fiber optic", internet == "DSL"], [70.0, 45.0], default=20.0
    )
    addons = sum(
        (df[c] == "Yes").to_numpy().astype(float)
        for c in [
            "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        ]
    )
    monthly = base + addons * rng.uniform(3.5, 6.5, size=n) + rng.normal(0, 2.5, size=n)
    monthly = np.clip(monthly, 18.25, 120.0).round(2)

    # TotalCharges drifts from tenure * monthly (price changes over the lifetime)
    total = (tenure * monthly * rng.uniform(0.92, 1.08, size=n)).round(2)

    df["MonthlyCharges"] = monthly
    df["TotalCharges"] = total.astype(object)

    # The real dataset stores TotalCharges as text with blanks for tenure == 0.
    # Reproduce that quirk so the cleaning code is exercised honestly.
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df["TotalCharges"] = df["TotalCharges"].astype(str)

    # ---- Churn: logit driven by the relationships the real dataset shows ----
    logit = (
        -0.78
        + 1.35 * (contract == "Month-to-month")
        - 0.85 * (contract == "Two year")
        - 0.045 * tenure
        + 0.55 * (payment == "Electronic check")
        + 0.60 * (internet == "Fiber optic")
        - 0.45 * (df["OnlineSecurity"] == "Yes").to_numpy()
        - 0.40 * (df["TechSupport"] == "Yes").to_numpy()
        + 0.30 * (paperless == "Yes")
        + 0.25 * df["SeniorCitizen"].to_numpy()
        - 0.20 * (df["Partner"] == "Yes").to_numpy()
        + 0.012 * (monthly - monthly.mean())
        + rng.normal(0, 0.55, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Telco churn data")
    parser.add_argument("--rows", type=int, default=7043)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = generate(args.rows, cfg.random_state)
    out = cfg.path("raw_csv")
    df.to_csv(out, index=False)

    rate = (df["Churn"] == "Yes").mean()
    print(f"Wrote {len(df):,} rows to {out}")
    print(f"Churn rate: {rate:.1%}  (real dataset is ~26.5%)")


if __name__ == "__main__":
    main()
