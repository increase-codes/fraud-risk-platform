
"""Load the trained fraud model and generate predictions."""

from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from fraud_risk.selection_config import SELECTED_THRESHOLD


MODEL_PATH = Path("artifacts/fraud_model.joblib")
THRESHOLD = SELECTED_THRESHOLD


@lru_cache(maxsize=1)
def load_model():
    """Load the saved model artifact once and reuse it."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH.resolve()}. "
            "Train the model first."
        )
    return joblib.load(MODEL_PATH)


def predict_fraud(data: pd.DataFrame, model=None) -> pd.DataFrame:
    """Return fraud probabilities and threshold-based predictions."""
    if data.empty:
        raise ValueError("Input data must contain at least one row.")

    if model is None:
        model = load_model()

    probabilities = model.predict_proba(data)[:, 1]

    return pd.DataFrame({
        "fraud_probability": probabilities,
        "fraud_prediction": (probabilities >= THRESHOLD).astype(int),
    })