"""Business logic: turning probabilities into money.

Kept separate from the modelling code because these are *assumptions*, not
statistics. Anyone reviewing the project should be able to find and challenge
them in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import Config


def customer_value(monthly_charges: np.ndarray, cfg: Config) -> np.ndarray:
    """Value at stake per customer over the configured horizon."""
    months = cfg["business"]["customer_lifetime_months"]
    return np.asarray(monthly_charges, dtype=float) * months


def revenue_at_risk(probabilities: np.ndarray, monthly_charges: np.ndarray, cfg: Config) -> float:
    """Expected revenue lost to churn = sum(P(churn) x customer value)."""
    return float(np.sum(np.asarray(probabilities) * customer_value(monthly_charges, cfg)))


def expected_value_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    monthly_charges: np.ndarray,
    threshold: float,
    cfg: Config,
) -> float:
    """Net uplift of a retention campaign at a threshold, versus doing nothing.

    We contact everyone scored at or above the threshold:
      true positive  -> pay the offer cost, retain them with probability `accept`
      false positive -> pay the offer cost for nothing
      false negative -> not contacted; they churn, exactly as they would have anyway

    Measuring *uplift against the do-nothing baseline* is the important detail.
    Churn losses from customers we never contact occur in the baseline too, so
    charging them against the campaign would double-count and push the optimiser
    toward flagging everybody. The algebra reduces to:

        uplift = (value retained) - (campaign spend)
    """
    b = cfg["business"]
    cost = float(b["retention_offer_cost"])
    accept = float(b["offer_acceptance_rate"])

    value = customer_value(monthly_charges, cfg)
    y_true = np.asarray(y_true).astype(bool)
    flagged = np.asarray(probabilities) >= threshold

    tp = flagged & y_true
    saved = float(np.sum(value[tp]) * accept)
    spend = float(cost * flagged.sum())

    return saved - spend


def tune_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    monthly_charges: np.ndarray,
    cfg: Config,
    grid: np.ndarray | None = None,
) -> tuple[float, float, pd.DataFrame]:
    """Sweep thresholds and return (best_threshold, best_uplift, full_curve).

    The default 0.5 cut-off is arbitrary - it assumes a false positive and a
    false negative cost the same, which is never true in retention. This picks
    the threshold maximising campaign uplift under the assumptions in
    config.yaml, and the curve is plotted in reports/figures/threshold_value.png
    so the choice is visible rather than asserted.
    """
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 2)

    rows = []
    for t in grid:
        rows.append(
            {
                "threshold": float(t),
                "expected_value": expected_value_at_threshold(
                    y_true, probabilities, monthly_charges, float(t), cfg
                ),
                "n_flagged": int((np.asarray(probabilities) >= t).sum()),
            }
        )
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["expected_value"].idxmax()]
    return float(best["threshold"]), float(best["expected_value"]), curve


def simulate_campaign(
    scored: pd.DataFrame,
    top_n: int,
    cfg: Config,
    prob_col: str = "churn_probability",
    charges_col: str = "MonthlyCharges",
) -> dict[str, float]:
    """'If we target the top N riskiest customers, what happens?'

    Uses predicted probability as the expected churn rate among those contacted,
    which is the honest thing to do when we do not know the outcome yet.
    """
    b = cfg["business"]
    cost = float(b["retention_offer_cost"])
    accept = float(b["offer_acceptance_rate"])

    target = scored.nlargest(top_n, prob_col)
    value = customer_value(target[charges_col].to_numpy(), cfg)
    prob = target[prob_col].to_numpy()

    campaign_cost = cost * len(target)
    expected_saved = float(np.sum(prob * value) * accept)

    return {
        "customers_targeted": int(len(target)),
        "campaign_cost": campaign_cost,
        "expected_revenue_saved": expected_saved,
        "net_benefit": expected_saved - campaign_cost,
        "roi": (expected_saved - campaign_cost) / campaign_cost if campaign_cost else 0.0,
        "avg_risk_of_targeted": float(prob.mean()) if len(target) else 0.0,
    }
