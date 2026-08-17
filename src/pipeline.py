"""Model pipeline construction.

The whole point of this module: preprocessing lives *inside* the estimator, so
the saved artifact is self-contained. Training and inference cannot drift apart,
and `handle_unknown="ignore"` means an unseen category in new data does not
crash scoring - it is encoded as all-zeros instead.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import Config


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore" is what makes the artifact reusable on new data.
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_estimators(cfg: Config) -> dict[str, object]:
    """Instantiate every enabled model from config."""
    mcfg = cfg["models"]
    rs = cfg.random_state
    estimators: dict[str, object] = {}

    if mcfg["logistic_regression"]["enabled"]:
        estimators["logistic_regression"] = LogisticRegression(
            random_state=rs, **mcfg["logistic_regression"]["params"]
        )
    if mcfg["random_forest"]["enabled"]:
        estimators["random_forest"] = RandomForestClassifier(
            random_state=rs, **mcfg["random_forest"]["params"]
        )
    if mcfg["gradient_boosting"]["enabled"]:
        estimators["gradient_boosting"] = HistGradientBoostingClassifier(
            random_state=rs, **mcfg["gradient_boosting"]["params"]
        )
    return estimators


def build_pipeline(estimator, numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(numeric_cols, categorical_cols)),
            ("clf", estimator),
        ]
    )


def feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Column names after one-hot encoding - needed for SHAP and coefficients."""
    return list(fitted_pipeline.named_steps["prep"].get_feature_names_out())
