"""Tests for the parts most likely to break silently.

Run:  pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from business import (  # noqa: E402
    expected_value_at_threshold,
    revenue_at_risk,
    simulate_campaign,
    tune_threshold,
)
from config import load_config  # noqa: E402
from features import add_engineered_features  # noqa: E402
from ingest import clean  # noqa: E402
from make_sample_data import generate  # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def raw(cfg):
    return generate(400, cfg.random_state)


@pytest.fixture(scope="module")
def feats(raw, cfg):
    return add_engineered_features(clean(raw), cfg)


# ------------------------------------------------------------------ cleaning
def test_clean_converts_blank_total_charges(raw):
    assert (raw["TotalCharges"].astype(str).str.strip() == "").any(), "fixture should contain blanks"
    out = clean(raw)
    assert pd.api.types.is_numeric_dtype(out["TotalCharges"])
    assert out["TotalCharges"].isna().sum() == 0


def test_clean_drops_duplicate_ids(raw):
    doubled = pd.concat([raw, raw.head(10)], ignore_index=True)
    out = clean(doubled)
    assert out["customerID"].duplicated().sum() == 0


# ------------------------------------------------------- feature engineering
def test_no_nulls_or_infinities(feats):
    assert feats.isna().sum().sum() == 0
    numeric = feats.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.to_numpy()).any()


def test_services_count_matches_manual_count(feats, cfg):
    cols = [c for c in cfg["features"]["service_columns"] if c in feats.columns]
    expected = (feats[cols] == "Yes").sum(axis=1)
    pd.testing.assert_series_equal(
        feats["services_count"], expected, check_names=False, check_dtype=False
    )


def test_zero_tenure_does_not_produce_infinite_spend(cfg):
    row = pd.DataFrame(
        {
            "customerID": ["X"], "tenure": [0], "TotalCharges": [0.0],
            "MonthlyCharges": [70.0], "Contract": ["Month-to-month"],
            "PaymentMethod": ["Electronic check"], "PhoneService": ["Yes"],
        }
    )
    out = add_engineered_features(row, cfg)
    assert np.isfinite(out["avg_monthly_spend"]).all()
    assert out["avg_monthly_spend"].iloc[0] == 70.0  # falls back to MonthlyCharges


def test_flags_are_binary(feats):
    for col in ["is_month_to_month", "has_auto_payment", "is_new_customer"]:
        assert set(feats[col].unique()) <= {0, 1}


def test_tenure_buckets_cover_every_row(feats, cfg):
    labels = set(cfg["features"]["tenure_buckets"]["labels"])
    assert set(feats["tenure_bucket"].unique()) <= labels
    assert "nan" not in feats["tenure_bucket"].unique()


# ------------------------------------------------------------------ business
def test_revenue_at_risk_scales_with_probability(cfg):
    charges = np.array([100.0, 100.0])
    low = revenue_at_risk(np.array([0.1, 0.1]), charges, cfg)
    high = revenue_at_risk(np.array([0.9, 0.9]), charges, cfg)
    assert high > low
    months = cfg["business"]["customer_lifetime_months"]
    assert low == pytest.approx(0.1 * 100 * months * 2)


def test_uplift_is_zero_when_nobody_is_flagged(cfg):
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.1, 0.1, 0.1, 0.1])
    charges = np.array([50.0] * 4)
    # Threshold above every score -> no contacts, no spend, no saving.
    assert expected_value_at_threshold(y, proba, charges, 0.99, cfg) == 0.0


def test_uplift_penalises_flagging_non_churners(cfg):
    """Flagging only true churners must beat flagging everyone."""
    y = np.array([1, 1, 0, 0, 0, 0])
    proba = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
    charges = np.array([100.0] * 6)
    precise = expected_value_at_threshold(y, proba, charges, 0.5, cfg)
    everyone = expected_value_at_threshold(y, proba, charges, 0.0, cfg)
    assert precise > everyone


def test_tuned_threshold_is_at_least_as_good_as_default(cfg):
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.27).astype(int)
    proba = np.clip(y * rng.normal(0.7, 0.15, n) + (1 - y) * rng.normal(0.25, 0.15, n), 0.01, 0.99)
    charges = rng.uniform(30, 110, n)
    threshold, value, curve = tune_threshold(y, proba, charges, cfg)
    assert 0.0 < threshold < 1.0
    assert value >= expected_value_at_threshold(y, proba, charges, 0.5, cfg)
    assert len(curve) > 10


def test_campaign_simulation_is_internally_consistent(cfg):
    scored = pd.DataFrame(
        {"churn_probability": np.linspace(0.01, 0.99, 200),
         "MonthlyCharges": np.full(200, 80.0)}
    )
    sim = simulate_campaign(scored, 50, cfg)
    assert sim["customers_targeted"] == 50
    assert sim["net_benefit"] == pytest.approx(
        sim["expected_revenue_saved"] - sim["campaign_cost"]
    )
    # Targeting the top 50 must select higher-risk customers than the average.
    assert sim["avg_risk_of_targeted"] > scored["churn_probability"].mean()


# ----------------------------------------------------------------- reusability
def test_unseen_category_does_not_break_encoding(feats, cfg):
    """The reason handle_unknown='ignore' is in the pipeline."""
    pytest.importorskip("sklearn")
    from features import split_columns
    from pipeline import build_pipeline
    from sklearn.linear_model import LogisticRegression

    numeric, categorical = split_columns(feats, cfg)
    X = feats[numeric + categorical]
    y = (feats["Churn"] == "Yes").astype(int)

    pipe = build_pipeline(LogisticRegression(max_iter=1000), numeric, categorical)
    pipe.fit(X, y)

    novel = X.head(5).copy()
    novel["Contract"] = "Five year"  # a value the model has never seen
    proba = pipe.predict_proba(novel)[:, 1]
    assert proba.shape == (5,)
    assert np.all((proba >= 0) & (proba <= 1))
