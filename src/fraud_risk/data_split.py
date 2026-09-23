
# ======== IMPORTS ========

# Import pandas for DataFrame operations.
import pandas as pd

# Import the project's centralized data-loading function.
from fraud_risk.data_loader import load_data


# ======== DATA SPLIT CONFIGURATION ========

# Define the months used to train the model.
TRAIN_MONTHS = {1, 2}

# Define the month used for model development and tuning.
VALIDATION_MONTH = 3

# Define the latest observed month reserved for final evaluation.
TEST_MONTH = 7


# ======== CHRONOLOGICAL DATA SPLITTING ========

def split_data(df):
    """
    Split the validated dataset into chronological partitions.

    The split is performed by month rather than randomly so that
    later observations do not enter model development.
    """

    # ======== EXPECTED MONTH VALIDATION ========

    # Define the complete set of month values expected by this
    # dataset version.
    expected_months = TRAIN_MONTHS | {
        VALIDATION_MONTH,
        TEST_MONTH,
    }

    # Extract the month values actually present in the dataset.
    actual_months = set(df["month"].unique())

    # Identify any month values that were not expected.
    unexpected_months = actual_months - expected_months

    # Fail explicitly if the dataset contains unexpected months.
    if unexpected_months:
        raise ValueError(
            f"Unexpected month values found: {sorted(unexpected_months)}"
        )

    # ======== DATA PARTITION CREATION ========

    # Create the training partition using the designated training months.
    train_df = df[df["month"].isin(TRAIN_MONTHS)].copy()

    # Create the validation partition using the designated validation month.
    validation_df = df[
        df["month"] == VALIDATION_MONTH
    ].copy()

    # Create the test partition using the latest reserved month.
    test_df = df[
        df["month"] == TEST_MONTH
    ].copy()

    # ======== PARTITION COMPLETENESS VALIDATION ========

    # Ensure that the training partition contains observations.
    if train_df.empty:
        raise ValueError("Training dataset is empty.")

    # Ensure that the validation partition contains observations.
    if validation_df.empty:
        raise ValueError("Validation dataset is empty.")

    # Ensure that the test partition contains observations.
    if test_df.empty:
        raise ValueError("Test dataset is empty.")

    # ======== PARTITION INTEGRITY VALIDATION ========

    # Convert partition indices to sets so that overlaps can be checked.
    train_indices = set(train_df.index)
    validation_indices = set(validation_df.index)
    test_indices = set(test_df.index)

    # Ensure that training and validation observations do not overlap.
    if train_indices & validation_indices:
        raise ValueError("Training and validation sets overlap.")

    # Ensure that training and test observations do not overlap.
    if train_indices & test_indices:
        raise ValueError("Training and test sets overlap.")

    # Ensure that validation and test observations do not overlap.
    if validation_indices & test_indices:
        raise ValueError("Validation and test sets overlap.")

    # Return the three validated chronological partitions.
    return train_df, validation_df, test_df


# ======== DATASET SPLIT REPORTING ========

def main():
    """Load the dataset, perform the split, and report the result."""

    # Load the validated dataset using the project's data loader.
    df = load_data()

    # Perform the chronological train/validation/test split.
    train_df, validation_df, test_df = split_data(df)

    # ======== TRAINING PARTITION REPORT ========

    print("Dataset split:")
    print()

    print("Training:")
    print("Rows:", len(train_df))
    print("Months:", sorted(train_df["month"].unique()))
    print(
        "Fraud rate:",
        round(train_df["fraud_bool"].mean() * 100, 4),
        "%",
    )

    # ======== VALIDATION PARTITION REPORT ========

    print()

    print("Validation:")
    print("Rows:", len(validation_df))
    print("Months:", sorted(validation_df["month"].unique()))
    print(
        "Fraud rate:",
        round(validation_df["fraud_bool"].mean() * 100, 4),
        "%",
    )

    # ======== TEST PARTITION REPORT ========

    print()

    print("Test:")
    print("Rows:", len(test_df))
    print("Months:", sorted(test_df["month"].unique()))
    print(
        "Fraud rate:",
        round(test_df["fraud_bool"].mean() * 100, 4),
        "%",
    )


# ======== MODULE ENTRY POINT ========

if __name__ == "__main__":
    main()
