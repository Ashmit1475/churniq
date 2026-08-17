# ChurnIQ

**A churn-risk scoring system that ranks telecom customers by predicted churn probability, explains each score, and quantifies the revenue at stake.**

Built with Python, SQL, scikit-learn, SHAP, Streamlit and Power BI.

> Replace this line with a GIF of the Streamlit app or Power BI report once you've recorded one — it is the first thing anyone looks at.

---

## The problem

Acquiring a telecom customer costs far more than keeping one, but retention teams can't call everybody. They need three things a spreadsheet can't give them:

1. **Who** is most likely to leave.
2. **Why** each of those customers is at risk, so the offer can be targeted.
3. **Whether** contacting them is worth the cost of the offer.

ChurnIQ answers all three. It scores every customer, attaches the top drivers behind each score via SHAP, and tunes its decision threshold to maximise the expected value of a retention campaign rather than defaulting to an arbitrary 0.5 cut-off.

## Architecture

```
 Raw CSV
    │
    ▼
 ingest.py ─────────► SQL database
    │                 ├── customers  (demographics, tenure, contract, services)
    │                 └── billing    (charges, payment method)
    ▼
 features.py ───────► JOIN + 7 engineered features ──► parquet
    │
    ▼
 train.py ──────────► 3 models compared (5-fold stratified CV)
    │                 └── best by ROC-AUC → threshold tuned for business value
    ▼
 models/churn_pipeline.pkl        (self-contained: preprocessing + model)
    │
    ▼
 predict.py ────────► predictions table in SQL  +  scored_customers.csv
    │                 └── probability, risk band, SHAP reasons, revenue at risk
    ▼
 ┌──────────────────┬──────────────────┐
 │  Power BI        │  Streamlit app   │
 │  (3-page report) │  (deployed)      │
 └──────────────────┴──────────────────┘
```

Training and inference are deliberately separate. `train.py` runs once and saves an artifact; `predict.py` loads that artifact and scores any file with the same schema. The model is never retrained at prediction time.

## Quickstart

```bash
git clone <your-repo-url> && cd churniq
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py          # generates data, ingests, trains, scores, plots
streamlit run src/app.py        # open the interactive app
```

That's it — the repo ships with a synthetic data generator, so it runs with no downloads and no credentials.

### Using the real dataset

The generator produces a **schema-identical synthetic** dataset (same 21 columns, same category values, realistic churn relationships) so the project is reproducible by anyone who clones it. To use the real data:

1. Download [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle.
2. Save it as `data/raw/Telco-Customer-Churn.csv`.
3. Run `python run_pipeline.py --skip-data`.

No code changes needed — `ingest.py` reads whichever CSV is at that path.

### Running stages individually

```bash
python src/make_sample_data.py --rows 7043   # synthetic data
python src/ingest.py                         # CSV  -> SQL tables (cleaned)
python src/features.py                       # SQL JOIN -> feature table
python src/train.py                          # compare, select, tune, save
python src/predict.py                        # score -> predictions table
python src/predict.py --csv new_batch.csv    # score a brand-new file
python src/evaluate.py                       # figures for the README
pytest -q                                    # tests
```

## Results

> Fill these in from `reports/metrics.json` after your first run. Do not invent numbers — you will be asked about them in an interview.

| Metric | Value |
|---|---|
| Rows | _7,043_ |
| Churn rate | _25.6%_ |
| Best model | _<from metrics.json>_ |
| Test ROC-AUC | _0.__ |
| Test PR-AUC | _0.__ |
| Precision @ tuned threshold | _0.__ |
| Recall @ tuned threshold | _0.__ |
| Tuned threshold | _0.__ (vs. 0.50 default) |
| Revenue at risk | _$___ |

**Model comparison** (5-fold stratified CV on the training split):

| Model | CV ROC-AUC | Test ROC-AUC |
|---|---|---|
| Logistic Regression | | |
| Random Forest | | |
| Gradient Boosting | | |

![Model comparison](reports/figures/model_comparison.png)
![ROC curve](reports/figures/roc_curve.png)
![Top churn drivers](reports/figures/churn_drivers.png)
![Threshold value](reports/figures/threshold_value.png)

## Design decisions worth explaining

**Why the data goes into SQL rather than staying in pandas.** The source CSV is flat; `ingest.py` splits it into `customers` and `billing` and the feature stage reads them back through a JOIN. This is an introduced split, not one inherent to the data — it exists so the project demonstrates schema design, joins and indexing rather than a single `read_csv`. Being upfront about that is better than pretending the source was normalised.

**Why accuracy is not the headline metric.** About 26% of customers churn, so a model that predicts "nobody churns" scores 74% accuracy and is worthless. The project reports ROC-AUC, PR-AUC, precision and recall, and shows the confusion matrix.

**Why the decision threshold isn't 0.5.** A 0.5 cut-off implicitly assumes a false positive and a false negative cost the same. In retention they don't: a false positive wastes one discount, a false negative loses a customer's remaining value. `business.py` sweeps every threshold and picks the one maximising campaign uplift versus doing nothing:

```
uplift = (value of retained customers) − (campaign spend)
```

Churn losses among customers we never contact are excluded deliberately — they happen in the do-nothing baseline too, so charging them against the campaign would double-count and push the optimiser toward flagging everybody. The full curve is plotted so the choice is visible rather than asserted.

**Why the model is a `Pipeline`, not a script of transformations.** Preprocessing lives inside the estimator, so the saved `.pkl` is self-contained and training and inference cannot drift apart. `OneHotEncoder(handle_unknown="ignore")` means an unseen category in new data is encoded as zeros instead of crashing — which is what makes the artifact reusable on any file with the same schema, not just the rows it was trained on. There's a test for exactly this.

**Why SHAP.** A risk score of 0.84 isn't actionable; "0.84 — month-to-month contract, 3 months tenure, electronic check" is. Every scored customer carries their top three risk drivers, and the Power BI call list surfaces them.

**Assumptions, stated plainly.** The offer cost, acceptance rate and 12-month value horizon in `config.yaml` are assumptions, not measurements. They drive the threshold and every currency figure in the project. Change them and the recommended threshold moves — the campaign planner in the Streamlit app is built to make that obvious.

## Engineered features

| Feature | Why |
|---|---|
| `tenure_bucket` | Churn risk is non-linear in tenure; buckets capture the early cliff |
| `avg_monthly_spend` | `TotalCharges / tenure` — detects spend drifting from the current bill |
| `services_count` | Bundling is strongly protective against churn |
| `is_month_to_month` | Typically the single strongest churn signal in this dataset |
| `has_auto_payment` | Electronic-check payers churn far more than auto-pay customers |
| `charge_ratio` | Current bill vs. lifetime average — a recent price-increase signal |
| `is_new_customer` | First-year customers behave differently from the rest |

## Project structure

```
churniq/
├── config.yaml                  # every path and parameter — nothing hardcoded in src/
├── run_pipeline.py              # one-command reproduction
├── requirements.txt
├── src/
│   ├── config.py                # config loading + path resolution
│   ├── db.py                    # SQLAlchemy, with a stdlib sqlite3 fallback
│   ├── make_sample_data.py      # schema-identical synthetic dataset
│   ├── ingest.py                # CSV -> cleaned, normalised SQL tables
│   ├── features.py              # SQL JOIN -> engineered feature table
│   ├── pipeline.py              # ColumnTransformer + estimator construction
│   ├── business.py              # revenue at risk, threshold tuning, campaign sim
│   ├── train.py                 # compare, cross-validate, tune, save artifact
│   ├── explain.py               # SHAP explanations (with graceful fallback)
│   ├── predict.py               # load artifact, score, write to SQL
│   ├── evaluate.py              # ROC / PR / confusion / drivers figures
│   └── app.py                   # Streamlit app
├── dashboard/
│   └── POWERBI_GUIDE.md         # DAX measures + 3-page build guide
├── tests/
│   └── test_pipeline.py         # 13 tests over cleaning, features, business logic
├── notebooks/
│   └── 01_eda.ipynb             # exploration — not the deliverable
└── reports/
    ├── metrics.json
    ├── scored_customers.csv
    └── figures/
```

## Deployment

**Streamlit app** — free on [Streamlit Community Cloud](https://share.streamlit.io): connect the GitHub repo, set the entrypoint to `src/app.py`. Commit `models/churn_pipeline.pkl` and `reports/scored_customers.csv` so the cloud instance has data without needing a database.

**Database** — SQLite works out of the box. For a hosted Postgres (Supabase or Neon, both free tier), set `DATABASE_URL` in `.env` and `pip install psycopg2-binary`; nothing else changes.

**Power BI** — `.pbix` is a binary format that can only be authored in Power BI Desktop, so it isn't generated by the pipeline. `dashboard/POWERBI_GUIDE.md` has the full build: connection setup, ~20 DAX measures, and the three-page layout. Commit the `.pbix` and screenshots.

## Limitations and what I'd do next

Being straight about these is deliberate — every one of them is a question an interviewer might ask.

- **The default dataset is synthetic.** Relationships are realistic and the schema matches exactly, but reported metrics should come from the real Kaggle file before you quote them anywhere.
- **No temporal validation.** The Telco dataset is a snapshot with no timestamps, so there's no train-on-past / test-on-future split. With real data I'd validate across time periods, because a random split leaks future information.
- **The business assumptions are guesses.** Offer cost and acceptance rate should come from an actual retention team, ideally measured by holdout testing.
- **No monitoring.** A production version needs drift detection on the input distribution and scheduled retraining.
- **The customers/billing split is synthetic**, as noted above.
- **Next:** a FastAPI endpoint for real-time scoring, a proper experiment log (MLflow) once there are more than a handful of runs, and an uplift model rather than a churn model — churn probability isn't the same as *persuadability*, and targeting the customers most likely to leave isn't the same as targeting the ones an offer would actually change.

## Tech stack

Python 3.10+ · pandas · scikit-learn · SQLAlchemy / SQLite / Postgres · SHAP · matplotlib · Plotly · Streamlit · Power BI · pytest

---

Built by [Ashmit Kuhikar](https://github.com/) — B.Tech CSE, SRM Institute of Science and Technology (2027).
