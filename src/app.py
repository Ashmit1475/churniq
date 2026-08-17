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
import pandas as pd
import plotly.express as px
import streamlit as st

from business import simulate_campaign
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

st.title("ChurnIQ")
st.caption("Churn-risk scoring and retention planning for a telecom subscriber base")

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
        c3.metric("Revenue at risk", f"{sym}{total_rar:,.0f}")
        c4.metric("Avg churn probability", f"{scored['churn_probability'].mean():.1%}")

        left, right = st.columns(2)
        with left:
            band_order = ["Low", "Medium", "High", "Critical"]
            counts = (
                scored["risk_band"].value_counts().reindex(band_order).fillna(0).reset_index()
            )
            counts.columns = ["Risk band", "Customers"]
            fig = px.bar(counts, x="Risk band", y="Customers", title="Customers by risk band")
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
                fig.update_yaxes(tickformat=".0%")
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
                title="Churn risk: contract type x tenure",
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
        min_value = f3.slider(
            f"Minimum customer value ({sym})", 0, 2000, 0, step=100
        )

        view = scored.copy()
        if bands:
            view = view[view["risk_band"].isin(bands)]
        if contracts:
            view = view[view["Contract"].isin(contracts)]
        if "customer_value" in view.columns:
            view = view[view["customer_value"] >= min_value]

        st.caption(
            f"{len(view):,} customers · "
            f"{sym}{view['revenue_at_risk'].sum():,.0f} revenue at risk"
        )
        display_cols = [
            c for c in [
                cfg["data"]["id_column"], "churn_probability", "risk_band",
                "MonthlyCharges", "tenure", "Contract", "revenue_at_risk",
                "top_risk_reasons",
            ] if c in view.columns
        ]
        st.dataframe(
            view[display_cols].head(500),
            use_container_width=True, hide_index=True,
            column_config={
                "churn_probability": st.column_config.ProgressColumn(
                    "Churn risk", min_value=0.0, max_value=1.0, format="%.0f%%"
                )
            },
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
        top_n = c1.slider("Customers to target", 50, min(3000, len(scored)), 500, step=50)
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
        m1.metric("Campaign cost", f"{sym}{sim['campaign_cost']:,.0f}")
        m2.metric("Expected revenue saved", f"{sym}{sim['expected_revenue_saved']:,.0f}")
        m3.metric("Net benefit", f"{sym}{sim['net_benefit']:,.0f}")
        m4.metric("ROI", f"{sim['roi']:.0%}")

        sweep = pd.DataFrame(
            [
                {"targeted": n, **simulate_campaign(scored, n, tweaked)}
                for n in range(50, min(3000, len(scored)) + 1, 50)
            ]
        )
        fig = px.line(
            sweep, x="customers_targeted", y="net_benefit",
            title="Net benefit by campaign size",
        )
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        st.plotly_chart(fig, use_container_width=True)
        best = sweep.loc[sweep["net_benefit"].idxmax()]
        st.info(
            f"Net benefit peaks at **{int(best['customers_targeted']):,} customers** "
            f"({sym}{best['net_benefit']:,.0f}). Beyond that, offers go to customers "
            "too unlikely to churn to justify the cost."
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
            c1, c2 = st.columns(2)
            c1.metric("High risk", f"{int((result['risk_flag'] == 'High').sum()):,}")
            c2.metric("Revenue at risk", f"{sym}{result['revenue_at_risk'].sum():,.0f}")
            st.dataframe(result.head(200), use_container_width=True, hide_index=True)
            st.download_button(
                "Download scored CSV",
                result.to_csv(index=False).encode(),
                "scored_customers.csv", "text/csv",
            )
        except Exception as exc:
            st.error(f"Could not score that file: {exc}")
