
"""
Run baseline model evaluation across classification thresholds.
"""

import argparse

import pandas as pd

from fraud_risk.data_loader import load_data
from fraud_risk.data_split import split_data
from fraud_risk.evaluation import (
    DEFAULT_THRESHOLDS,
    evaluate_model,
    evaluate_thresholds,
)
from fraud_risk.model import build_lightgbm_model, build_model


def main():
    """Load data, train the selected model, and evaluate validation results."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fraud detection model."
    )

    parser.add_argument(
        "--model",
        choices=["logistic", "lightgbm"],
        default="logistic",
        help="Model to evaluate.",
    )

    args = parser.parse_args()

    # Data
    df = load_data()
    train_df, validation_df, _ = split_data(df)

    X_train = train_df.drop(columns=["fraud_bool", "month"])
    y_train = train_df["fraud_bool"]

    X_validation = validation_df.drop(columns=["fraud_bool", "month"])
    y_validation = validation_df["fraud_bool"]

    # Model selection
    if args.model == "lightgbm":
        model = build_lightgbm_model()
        model_name = "LightGBM"
    else:
        model = build_model()
        model_name = "Logistic Regression"

    model.fit(X_train, y_train)

    # Overall validation metrics
    metrics = evaluate_model(
        model,
        X_validation,
        y_validation,
    )

    print(f"{model_name} validation metrics:")
    print()
    print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

    # Threshold evaluation
    results = evaluate_thresholds(
        model,
        X_validation,
        y_validation,
        DEFAULT_THRESHOLDS,
    )

    results_df = pd.DataFrame(results)

    high_recall_df = results_df[
        results_df["recall"] >= 0.70
    ].copy()

    print()
    print("Thresholds with recall >= 70%:")

    print(
        high_recall_df[
            [
                "threshold",
                "precision",
                "recall",
                "alerts",
                "false_positives",
                "false_negatives",
            ]
        ].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
            },
        )
    )


if __name__ == "__main__":
    main()