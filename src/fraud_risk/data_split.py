
"""Split fraud data into chronological training, validation, and test sets."""

import pandas as pd

from fraud_risk.data_loader import load_data


# Data split configuration
TRAIN_MONTHS = {1, 2}
VALIDATION_MONTH = 3
TEST_MONTH = 7


def split_data(df):
    """Split data chronologically and validate partition integrity."""

    expected_months = TRAIN_MONTHS | {VALIDATION_MONTH, TEST_MONTH}

    # Validate the month column.
    if "month" not in df.columns:
        raise ValueError("Dataset must contain a 'month' column.")

    if df["month"].isna().any():
        raise ValueError("Month column contains missing values.")

    actual_months = set(df["month"].unique())

    unexpected_months = actual_months - expected_months
    if unexpected_months:
        raise ValueError(
            f"Unexpected month values found: {sorted(unexpected_months)}"
        )

    # Create chronological partitions.
    train_df = df[df["month"].isin(TRAIN_MONTHS)].copy()
    validation_df = df[df["month"] == VALIDATION_MONTH].copy()
    test_df = df[df["month"] == TEST_MONTH].copy()

    # Check partition completeness first for clear error messages.
    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    # Ensure all expected months are represented.
    missing_months = expected_months - actual_months
    if missing_months:
        raise ValueError(
            f"Missing expected month values: {sorted(missing_months)}"
        )

    # Ensure partitions contain no overlapping rows.
    train_indices = set(train_df.index)
    validation_indices = set(validation_df.index)
    test_indices = set(test_df.index)

    if train_indices & validation_indices:
        raise ValueError("Training and validation sets overlap.")

    if train_indices & test_indices:
        raise ValueError("Training and test sets overlap.")

    if validation_indices & test_indices:
        raise ValueError("Validation and test sets overlap.")

    return train_df, validation_df, test_df


def main():
    """Load the dataset, split it, and report partition statistics."""

    df = load_data()
    train_df, validation_df, test_df = split_data(df)

    print("Dataset split:\n")

    partitions = [
        ("Training", train_df),
        ("Validation", validation_df),
        ("Test", test_df),
    ]

    for name, partition in partitions:
        print(f"{name}:")
        print("Rows:", len(partition))
        print("Months:", sorted(partition["month"].unique()))
        print(
            "Fraud rate:",
            round(partition["fraud_bool"].mean() * 100, 4),
            "%",
        )
        print()


if __name__ == "__main__":
    main()