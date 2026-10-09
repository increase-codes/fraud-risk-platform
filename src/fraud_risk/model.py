"""
Model pipelines for fraud detection.

Provides builders for the Logistic Regression baseline
and the LightGBM gradient-boosted tree model.
"""

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from fraud_risk.preprocessing import build_preprocessor


# --- Logistic Regression configuration ----------------------------------

LOGISTIC_REGRESSION_CONFIG = {
    "C": 1.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
}


# --- LightGBM configuration ---------------------------------------------

LIGHTGBM_CONFIG = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}


# --- Logistic Regression model ------------------------------------------

def build_model():
    """Return a fresh preprocessing + Logistic Regression pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(**LOGISTIC_REGRESSION_CONFIG)),
    ])


# --- LightGBM model -----------------------------------------------------

def build_lightgbm_model():
    """Return a fresh preprocessing + LightGBM pipeline."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LGBMClassifier(**LIGHTGBM_CONFIG)),
    ])