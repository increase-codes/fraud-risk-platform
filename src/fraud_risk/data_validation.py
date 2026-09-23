# Import the function that loads our raw dataset.
import pandas as pd
from fraud_risk.data_loader import load_data


# ============================================================
# VALIDATION FUNCTION
# ============================================================
# This function receives a DataFrame and checks whether
# the data satisfies the rules our ML system expects.
def validate_schema(df):

    # ========================================================
    # 1. INSPECT DATASET
    # ========================================================
    # Basic information: size, columns, and data types.

    print("Shape:", df.shape)

    print("Columns:", df.columns.tolist())

    print("Data types:")
    print(df.dtypes)


    # --------------------------------------------------------
    # Check for missing values.
    # --------------------------------------------------------
    print()
    print("Missing values:")
    print(df.isna().sum())


    # --------------------------------------------------------
    # Check for completely duplicated rows.
    # --------------------------------------------------------
    print()
    print("Duplicate rows:", df.duplicated().sum())


    # --------------------------------------------------------
    # Inspect the target variable.
    # --------------------------------------------------------
    print()
    print("Target values:")
    print(df["fraud_bool"].value_counts().sort_index())


    # --------------------------------------------------------
    # Inspect important numeric feature ranges.
    # --------------------------------------------------------
    print()
    print("Customer age range:")
    print("Minimum:", df["customer_age"].min())
    print("Maximum:", df["customer_age"].max())

    print()
    print("Income range:")
    print("Minimum:", df["income"].min())
    print("Maximum:", df["income"].max())

    print()
    print("Name-email similarity range:")
    print("Minimum:", df["name_email_similarity"].min())
    print("Maximum:", df["name_email_similarity"].max())


    # --------------------------------------------------------
    # Inspect categorical feature values.
    # --------------------------------------------------------
    print()
    print("Payment types:")
    print(df["payment_type"].value_counts())

    print()
    print("Employment status:")
    print(df["employment_status"].value_counts())

    print()
    print("Housing status:")
    print(df["housing_status"].value_counts())

    print()
    print("Source:")
    print(df["source"].value_counts())

    print()
    print("Device OS:")
    print(df["device_os"].value_counts())


    # --------------------------------------------------------
    # Inspect a feature containing a sentinel value (-1).
    # We investigate it rather than automatically changing it.
    # --------------------------------------------------------
    print()
    print("Previous address months range:")
    print("Minimum:", df["prev_address_months_count"].min())
    print("Maximum:", df["prev_address_months_count"].max())


    # --------------------------------------------------------
    # Inspect count-based feature.
    # --------------------------------------------------------
    print()
    print("DOB distinct emails range:")
    print("Minimum:", df["date_of_birth_distinct_emails_4w"].min())
    print("Maximum:", df["date_of_birth_distinct_emails_4w"].max())
    print("Unique:", df["date_of_birth_distinct_emails_4w"].nunique())

    # --------------------------------------------------------
    # Inspect month distribution.
    # --------------------------------------------------------
    print()
    print("Month distribution:")
    print(df["month"].value_counts().sort_index())

    print()
    print("Fraud rate by month:")
    print(df.groupby("month")["fraud_bool"].mean() * 100)

    print()
    print("Fraud counts by month:")
    print(pd.crosstab(df["month"], df["fraud_bool"]))


    

    # ========================================================
    # 2. BASIC VALIDATION
    # ========================================================
    # Convert what we learned during inspection into rules.
    #
    # assert = "this condition MUST be true".
    # If false -> Python stops with an error.

    print()
    print("Validation checks:")

    # Target must contain only 0 and 1.
    assert set(df["fraud_bool"].unique()).issubset({0, 1})

    # Income must stay inside the observed/expected range.
    assert df["income"].between(0.1, 0.9).all()

    # Customer age must stay inside the defined range.
    assert df["customer_age"].between(10, 90).all()

    # Similarity score must be between 0 and 1.
    assert df["name_email_similarity"].between(0, 1).all()

    print("Basic validation: PASSED")



    # ========================================================
    # 3. REQUIRED COLUMNS
    # ========================================================
    # Define the columns our system requires.

    required_columns = {
        "fraud_bool",
        "income",
        "name_email_similarity",
        "customer_age",
        "payment_type",
        "employment_status",
        "housing_status",
        "source",
        "device_os",
    }

    # Every required column must exist in the DataFrame.
    assert required_columns.issubset(df.columns)

    print("Required columns: PASSED")


    # ========================================================
    # 4. FINAL AUTOMATED RULES
    # ========================================================

    # --------------------------------------------------------
    # 4A. CATEGORICAL VALUES
    # --------------------------------------------------------
    # These are the values our dataset is expected to contain.

    expected_categories = {
        "payment_type": {"AA", "AB", "AC", "AD", "AE"},
        "source": {"INTERNET", "TELEAPP"},
        "device_os": {"linux", "windows", "macintosh", "x11", "other"},
    }

    # Check every categorical column against its allowed values.
    for column, allowed_values in expected_categories.items():
        assert set(df[column].unique()).issubset(allowed_values)


    # --------------------------------------------------------
    # 4B. BINARY COLUMNS
    # --------------------------------------------------------
    # These features are flags, therefore they must contain
    # only 0 or 1.

    binary_columns = [
        "email_is_free",
        "phone_home_valid",
        "phone_mobile_valid",
        "has_other_cards",
        "foreign_request",
        "keep_alive_session",
    ]

    # Check every binary column.
    for column in binary_columns:
        assert set(df[column].unique()).issubset({0, 1})


    # --------------------------------------------------------
    # 4C. MISSING VALUES
    # --------------------------------------------------------
    # The dataset currently contains no missing values.
    # If missing values appear, validation stops.

    assert df.isna().sum().sum() == 0


    # --------------------------------------------------------
    # 4D. DUPLICATES
    # --------------------------------------------------------
    # The dataset currently contains no completely duplicated rows.

    assert df.duplicated().sum() == 0


    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------
    print("Categorical values: PASSED")
    print("Binary columns: PASSED")
    print("Missing values: PASSED")
    print("Duplicate rows: PASSED")
    print("All validation checks: PASSED")


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================
# This is what actually starts the validation when we run:
#
# python -m fraud_risk.data_validation
#

def main():

    # Load the raw dataset.
    df = load_data()

    # Send the dataset through our validation function.
    validate_schema(df)


# Only execute main() when this file is run directly.
if __name__ == "__main__":
    main()