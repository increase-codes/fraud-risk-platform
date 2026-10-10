
"""Evaluate the selected fraud model on the reserved test month."""

from fraud_risk.data_loader import load_data
from fraud_risk.data_split import split_data
from fraud_risk.evaluation import evaluate_model, evaluate_thresholds
from fraud_risk.model import build_lightgbm_model
from fraud_risk.selection_config import (
    SELECTED_THRESHOLD,
    TEST_MONTH,
)


def main():
    """Train on months 1–3 and evaluate once on month 7."""
    df = load_data()
    train_df, validation_df, test_df = split_data(df)

    # Confirm the reserved test partition matches our configuration.
    actual_test_months = set(test_df["month"].unique())
    if actual_test_months != {TEST_MONTH}:
        raise ValueError(
            f"Expected test month {TEST_MONTH}, "
            f"found {sorted(actual_test_months)}"
        )

    # After model and threshold selection, combine months 1–3
    # for final training. Month 7 remains excluded from fitting.
    development_df = df[df["month"].isin([1, 2, 3])].copy()

    feature_columns = [
        column for column in development_df.columns
        if column not in {"fraud_bool", "month"}
    ]

    X_train = development_df[feature_columns]
    y_train = development_df["fraud_bool"]

    X_test = test_df[feature_columns]
    y_test = test_df["fraud_bool"]

    model = build_lightgbm_model()
    model.fit(X_train, y_train)

    # Report ranking metrics and threshold-dependent metrics.
    metrics = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=SELECTED_THRESHOLD,
    )

    print("FINAL TEST RESULTS")
    print(f"Training months: {[1, 2, 3]}")
    print(f"Test month: {TEST_MONTH}")
    print(f"Threshold: {SELECTED_THRESHOLD:.2f}")
    print(f"Test rows: {len(test_df)}")
    print(f"Test fraud rate: {y_test.mean():.4%}")
    print()
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    # Confusion matrix uses [[TN, FP], [FN, TP]].
    tn, fp, fn, tp = metrics["confusion_matrix"].ravel()
    print()
    print(f"True negatives:  {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives:  {tp}")
    print(f"Total alerts:    {fp + tp}")


if __name__ == "__main__":
    main()