"""ChurnIQ Streamlit app.

Run locally:
    streamlit run src/app.py

Deploy free on Streamlit Community Cloud: push the repo to GitHub, point the
app at src/app.py, and commit models/churn_pipeline.pkl plus
reports/scored_customers.csv so the cloud instance has data without a database.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from business import campaign_sweep, simulate_campaign
from config import load_config
from ingest import clean
from features import add_engineered_features
from predict import score

st.set_page_config(page_title="ChurnIQ", page_icon="●", layout="wide")


@st.cache_resource
def get_config():
    return load_config()


@st.cache_resource
def get_artifact(_cfg):
    """cache_resource: the model is loaded once per session, not per interaction."""
    path = _cfg.path("model_artifact")
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_scored(_cfg):
    """cache_data: the dataframe is cached and invalidated by Streamlit's hashing."""
    csv = _cfg.path("scored_csv")
    if csv.exists():
        return pd.read_csv(csv)
    try:
        from db import get_engine, read_sql

        return read_sql(f"SELECT * FROM {_cfg['database']['predictions_table']}", get_engine(_cfg))
    except Exception:
        return None


cfg = get_config()
artifact = get_artifact(cfg)
scored = load_scored(cfg)
sym = cfg["business"]["currency_symbol"]


def money(amount: float) -> str:
    """Currency with the sign outside the symbol: -$22,345, not $-22,345."""
    return f"-{sym}{abs(amount):,.0f}" if amount < 0 else f"{sym}{amount:,.0f}"


st.title("ChurnIQ")
st.caption("Churn-risk scoring and retention planning for a telecom subscriber base")
st.markdown(
    "Every subscriber here is scored for how likely they are to cancel, and that score "
    "is turned into money — who to contact, what a retention campaign costs, and what "
    "it saves. The figures come from the public *Telco Customer Churn* dataset"
    f"{f' ({len(scored):,} subscribers)' if scored is not None else ''}, a published "
    "benchmark rather than live customer data."
)

if artifact is None:
    st.error(
        "No model artifact found. Run the pipeline first:\n\n"
        "```\npython src/make_sample_data.py\npython src/ingest.py\n"
        "python src/train.py\npython src/predict.py\n```"
    )
    st.stop()

st.sidebar.markdown(f"**Model:** {artifact['model_name']}")
st.sidebar.markdown(f"**Threshold:** {artifact['threshold']:.2f}")
st.sidebar.markdown(f"**Trained:** {artifact['trained_at'][:10]}")
st.sidebar.caption(
    "**What the threshold means.** A customer is flagged High risk at a churn "
    f"probability of {artifact['threshold']:.0%} or above. That cut-off replaces the "
    "usual 50% default: it was tuned to maximise the net benefit of a retention "
    f"campaign, assuming a {money(cfg['business']['retention_offer_cost'])} offer that "
    f"is accepted {cfg['business']['offer_acceptance_rate']:.0%} of the time. Missing a "
    "churner costs far more than wasting an offer, so the cut-off sits where the two "
    "errors balance out in money rather than in accuracy."
)

# Shared between the risk table and the scorer so both read identically.
RISK_COLUMN_CONFIG = {
    "churn_probability": st.column_config.ProgressColumn(
        "Churn risk", min_value=0.0, max_value=1.0, format="%.0f%%"
    )
}


def risk_display_cols(df: pd.DataFrame) -> list[str]:
    """The business-facing columns, in reading order - not the model plumbing."""
    return [
        c for c in [
            cfg["data"]["id_column"], "churn_probability", "risk_band",
            "MonthlyCharges", "tenure", "Contract", "revenue_at_risk",
            "top_risk_reasons",
        ] if c in df.columns
    ]


tab_overview, tab_risk, tab_campaign, tab_scorer = st.tabs(
    ["Overview", "Customer risk", "Campaign planner", "Score new data"]
)

# ---------------------------------------------------------------- Overview
with tab_overview:
    if scored is None:
        st.warning("No scored customers yet - run `python src/predict.py`.")
    else:
        total_rar = scored["revenue_at_risk"].sum()
        n_high = int((scored["risk_flag"] == "High").sum())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{len(scored):,}")
        c2.metric("High risk", f"{n_high:,}", f"{n_high / len(scored):.1%} of base")
        c3.metric("Revenue at risk", money(total_rar))
        c4.metric("Avg churn probability", f"{scored['churn_probability'].mean():.1%}")
        st.caption(
            '"High risk" means a churn probability at or above the tuned '
            f"{artifact['threshold']:.0%} threshold — see the sidebar for what that "
            'cut-off is optimising. "Revenue at risk" is each customer\'s churn '
            f"probability times their {cfg['business']['customer_lifetime_months']}-month "
            "value, summed across the base."
        )

        left, right = st.columns(2)
        with left:
            band_order = ["Low", "Medium", "High", "Critical"]
            counts = (
                scored["risk_band"].value_counts().reindex(band_order).fillna(0).reset_index()
            )
            counts.columns = ["Risk band", "Customers"]
            fig = px.bar(
                counts, x="Risk band", y="Customers",
                title="Customers by risk band",
                category_orders={"Risk band": band_order},
            )
            fig.update_xaxes(title_text="Risk band")
            fig.update_yaxes(title_text="Customers", tickformat=",d")
            fig.update_traces(
                hovertemplate="%{x} risk<br>%{y:,} customers<extra></extra>"
            )
            st.plotly_chart(fig, use_container_width=True)
        with right:
            if "Contract" in scored.columns:
                by_contract = (
                    scored.groupby("Contract")["churn_probability"].mean().reset_index()
                )
                fig = px.bar(
                    by_contract, x="Contract", y="churn_probability",
                    title="Average churn probability by contract type",
                )
                fig.update_xaxes(title_text="Contract type")
                fig.update_yaxes(title_text="Average churn probability", tickformat=".0%")
                fig.update_traces(
                    hovertemplate="%{x}<br>%{y:.1%} average churn probability<extra></extra>"
                )
                st.plotly_chart(fig, use_container_width=True)

        if {"tenure_bucket", "Contract"}.issubset(scored.columns):
            heat = (
                scored.pivot_table(
                    index="Contract", columns="tenure_bucket",
                    values="churn_probability", aggfunc="mean",
                )
                .reindex(columns=cfg["features"]["tenure_buckets"]["labels"])
            )
            fig = px.imshow(
                heat, text_auto=".1%", aspect="auto", color_continuous_scale="Reds",
                title="Average churn probability: contract type × tenure",
                labels={
                    "x": "Tenure band", "y": "Contract type",
                    "color": "Average churn probability",
                },
            )
            fig.update_coloraxes(colorbar_tickformat=".0%")
            fig.update_traces(
                hovertemplate=(
                    "%{y} contract · %{x} tenure<br>"
                    "%{z:.1%} average churn probability<extra></extra>"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------ Customer risk
with tab_risk:
    if scored is None:
        st.warning("Run `python src/predict.py` first.")
    else:
        f1, f2, f3 = st.columns(3)
        bands = f1.multiselect(
            "Risk band", ["Critical", "High", "Medium", "Low"], default=["Critical", "High"]
        )
        contracts = f2.multiselect(
            "Contract",
            sorted(scored["Contract"].dropna().unique()) if "Contract" in scored else [],
            default=None,
        )
        months = cfg["business"]["customer_lifetime_months"]
        if "customer_value" in scored.columns:
            # Bound the slider by the data - a fixed 0-2000 leaves most of the
            # track dead when the real maximum is far below it.
            value_cap = int(np.ceil(scored["customer_value"].max() / 100.0) * 100)
            min_value = f3.slider(
                f"Minimum {months}-month customer value ({sym})",
                0, value_cap, 0, step=100,
            )
        else:
            min_value = 0

        view = scored.copy()
        if bands:
            view = view[view["risk_band"].isin(bands)]
        if contracts:
            view = view[view["Contract"].isin(contracts)]
        if "customer_value" in view.columns:
            view = view[view["customer_value"] >= min_value]

        if not bands:
            st.caption("No risk band selected — showing every band.")
        st.caption(
            f"{len(view):,} customers · "
            f"{money(view['revenue_at_risk'].sum())} revenue at risk"
        )
        display_cols = risk_display_cols(view)
        TABLE_LIMIT = 500
        st.dataframe(
            view[display_cols].head(TABLE_LIMIT),
            use_container_width=True, hide_index=True,
            column_config=RISK_COLUMN_CONFIG,
        )
        if len(view) > TABLE_LIMIT:
            st.caption(
                f"Showing the {TABLE_LIMIT} riskiest of {len(view):,} matching "
                "customers. The download below contains all of them."
            )
        st.download_button(
            "Download this list as CSV",
            view[display_cols].to_csv(index=False).encode(),
            "churn_risk_list.csv", "text/csv",
        )

# --------------------------------------------------------- Campaign planner
with tab_campaign:
    if scored is None:
        st.warning("Run `python src/predict.py` first.")
    else:
        st.markdown(
            "Estimate the return on a retention campaign targeting the riskiest "
            "customers. Adjust the assumptions and watch the economics change."
        )
        c1, c2, c3 = st.columns(3)
        top_n = c1.slider(
            "Customers to target", 50, len(scored), min(500, len(scored)), step=50
        )
        cost = c2.number_input(
            f"Offer cost per customer ({sym})", 10.0, 500.0,
            float(cfg["business"]["retention_offer_cost"]), step=10.0,
        )
        accept = c3.slider(
            "Offer acceptance rate", 0.05, 0.90,
            float(cfg["business"]["offer_acceptance_rate"]), step=0.05,
        )

        tweaked = load_config()
        tweaked["business"]["retention_offer_cost"] = cost
        tweaked["business"]["offer_acceptance_rate"] = accept
        sim = simulate_campaign(scored, top_n, tweaked)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Campaign cost", money(sim["campaign_cost"]))
        m2.metric("Expected revenue saved", money(sim["expected_revenue_saved"]))
        m3.metric("Net benefit", money(sim["net_benefit"]))
        m4.metric("ROI", f"{sim['roi']:.0%}")

        # Sweep the whole base, not a truncated slice: capping the sweep puts the
        # maximum on the last point evaluated and reports a peak that isn't one.
        sweep = campaign_sweep(scored, tweaked)
        fig = px.line(
            sweep, x="customers_targeted", y="net_benefit",
            title="Net benefit by campaign size",
            labels={
                "customers_targeted": "Customers targeted",
                "net_benefit": f"Net benefit ({sym})",
            },
        )
        fig.update_xaxes(title_text="Customers targeted (riskiest first)", tickformat=",d")
        fig.update_yaxes(
            title_text=f"Net benefit ({sym})", tickprefix=sym, tickformat=",.0f"
        )
        fig.update_traces(
            hovertemplate=(
                "Target the top %{x:,} customers<br>"
                f"Net benefit: {sym}" "%{y:,.0f}<extra></extra>"
            )
        )
        fig.add_hline(
            y=0, line_dash="dot", line_color="grey",
            annotation_text="break-even", annotation_position="bottom right",
        )
        fig.add_vline(
            x=top_n, line_dash="dash", line_color="#636efa",
            annotation_text="your campaign", annotation_position="top",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Each point answers: if we contacted the N riskiest customers, what would "
            "we net? Expected revenue saved minus the cost of every offer sent — "
            "including the ones that go to customers who were never going to leave."
        )

        best = sweep.loc[sweep["net_benefit"].idxmax()]
        best_n = int(best["customers_targeted"])
        best_net = float(best["net_benefit"])
        largest_evaluated = int(sweep["customers_targeted"].iloc[-1])

        if best_net <= 0:
            # The maximum is still a loss - reporting it as a "peak" would read as
            # a recommendation to run a campaign that never pays for itself.
            st.warning(
                f"**No campaign size breaks even** under these assumptions. The best "
                f"case is still a loss of {money(abs(best_net))} at {best_n:,} customers. "
                f"At {money(cost)} per offer and a {accept:.0%} acceptance rate, the "
                "offer costs more than the churn it prevents is worth — lower the offer "
                "cost or raise the acceptance rate before running anything."
            )
        elif best_n >= largest_evaluated:
            st.info(
                f"Net benefit is **still rising at {best_n:,} customers** — the entire "
                f"scored base ({money(best_net)}). Under these assumptions every "
                "additional offer still pays for itself, so there is no optimal cut-off "
                "to read off this curve. Raise the offer cost or lower the acceptance "
                "rate to find a turning point."
            )
        else:
            st.info(
                f"Net benefit peaks at **{best_n:,} customers** ({money(best_net)}). "
                "Beyond that, offers go to customers too unlikely to churn to justify "
                "the cost."
            )

# ------------------------------------------------------------ Score new data
with tab_scorer:
    st.markdown(
        "Upload a CSV with the same schema as the training data. The saved "
        "pipeline handles cleaning, feature engineering and encoding, so any file "
        "with the same columns works - not just the rows the model was trained on."
    )
    upload = st.file_uploader("Customer CSV", type="csv")
    if upload is not None:
        try:
            raw = clean(pd.read_csv(upload))
            feats = add_engineered_features(raw, cfg)
            result = score(feats, artifact, cfg)
            st.success(f"Scored {len(result):,} customers.")
            c1, c2, c3 = st.columns(3)
            c1.metric("High risk", f"{int((result['risk_flag'] == 'High').sum()):,}")
            c2.metric("Revenue at risk", money(result["revenue_at_risk"].sum()))
            c3.metric("Avg churn probability", f"{result['churn_probability'].mean():.1%}")

            PREVIEW_ROWS = 200
            st.dataframe(
                result[risk_display_cols(result)].head(PREVIEW_ROWS),
                use_container_width=True, hide_index=True,
                column_config=RISK_COLUMN_CONFIG,
            )
            if len(result) > PREVIEW_ROWS:
                st.caption(
                    f"Preview of the {PREVIEW_ROWS} riskiest of {len(result):,} scored "
                    "customers. The download below contains every row and every column."
                )
            st.download_button(
                "Download scored CSV",
                result.to_csv(index=False).encode(),
                "scored_customers.csv", "text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score that file: {exc}")
