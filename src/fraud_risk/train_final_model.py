
"""Train and package the selected fraud detection model."""

from pathlib import Path

import joblib
import pandas as pd

from fraud_risk.data_loader import load_data
from fraud_risk.data_split import split_data
from fraud_risk.model import build_lightgbm_model
from fraud_risk.preprocessing import TARGET_COLUMN, TIME_COLUMN
from fraud_risk.selection_config import SELECTED_MODEL


# Artifact configuration
MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"


def train_final_model():
    """Fit the selected pipeline on training and validation data."""

    if SELECTED_MODEL != "lightgbm":
        raise ValueError(
            f"Unsupported selected model: {SELECTED_MODEL}"
        )

    df = load_data()
    train_df, validation_df, _ = split_data(df)

    development_df = pd.concat(
        [train_df, validation_df],
        axis=0,
    )

    X = development_df.drop(
        columns=[TARGET_COLUMN, TIME_COLUMN]
    )
    y = development_df[TARGET_COLUMN]

    model = build_lightgbm_model()
    model.fit(X, y)

    return model


def main():
    """Train the model and save its artifact."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model = train_final_model()
    joblib.dump(model, MODEL_PATH)

    print(f"Selected model: {SELECTED_MODEL}")
    print(f"Artifact saved: {MODEL_PATH}")


if __name__ == "__main__":
    main()