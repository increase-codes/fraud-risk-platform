
"""Train and validate a candidate without changing the active model."""

from pathlib import Path

import joblib

from fraud_risk.data_loader import load_data
from fraud_risk.data_split import split_data
from fraud_risk.evaluation import evaluate_model
from fraud_risk.model import build_lightgbm_model
from fraud_risk.selection_config import MINIMUM_RECALL, SELECTED_THRESHOLD

CANDIDATE_PATH = Path("artifacts/fraud_model_candidate.joblib")


def train_candidate():
    """Train on months 1-2 and evaluate on validation month 3."""
    df = load_data()
    train_df, validation_df, _ = split_data(df)

    X_train = train_df.drop(columns=["fraud_bool", "month"])
    y_train = train_df["fraud_bool"]

    X_validation = validation_df.drop(columns=["fraud_bool", "month"])
    y_validation = validation_df["fraud_bool"]

    model = build_lightgbm_model()
    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
        threshold=SELECTED_THRESHOLD,
    )
    return model, metrics


def main():
    """Save the candidate only when validation recall meets the requirement."""
    model, metrics = train_candidate()

    print(f"Validation PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Validation recall: {metrics['recall']:.4f}")
    print(f"Required recall:   {MINIMUM_RECALL:.4f}")

    if metrics["recall"] < MINIMUM_RECALL:
        print("REJECTED: candidate did not meet the recall requirement.")
        return

    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, CANDIDATE_PATH)

    print(f"ACCEPTED: candidate saved to {CANDIDATE_PATH}")
    print("Active model unchanged; deployment requires a separate step.")


if __name__ == "__main__":
    main()
