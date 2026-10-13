# Fraud Risk Platform — Project Handoff

> Generated: 2026-10-09 16:30:48 +01:00
> Project root: `C:\Users\HP\Documents\#ML_ENGINE\fraud-risk-platform`
> This is a read-only reconnaissance report, not a certification that the project works.

## 1. Git State

**Inside Git repository:** true
**Branch:** main

### Working tree status
```text
## main...origin/main
?? PROJECT_CONTINUATION_REPORT.md
?? src/fraud_risk/run_final_evaluation.py
?? src/fraud_risk/train_final_model.py
```

### Latest 15 commits
```text
3a4adee 2026-10-09 Add final fraud model evaluation report
e1bc7b9 2026-10-09 Add fraud model evaluation and selection baseline
eb74594 2026-09-26 Add leakage-safe preprocessing pipeline
e5fa999 2026-09-23 Add date obtained, redistribution note, and hash verification steps to data provenance
d38e973 2026-09-23 Add chronological data split with validation checks
7f353fc 2026-09-09 Initial project structure
```

### Remotes (credentials redacted)
```text
origin	https://github.com/increase-codes/fraud-risk-platform.git (fetch)
origin	https://github.com/increase-codes/fraud-risk-platform.git (push)
```

### Last commit details
```text
3a4adee Add final fraud model evaluation report
 reports/final_test_results.md | 53 +++++++++++++++++++++++++++++++++++++++++++
 1 file changed, 53 insertions(+)
```

### Uncommitted diff summary
```text


```

## 2. File Inventory

Scanned files: **25**

| File | Size (bytes) | Last modified |
|---|---:|---|
| `.gitignore` | 150 | 2026-09-23 12:52 |
| `data\raw\CCFraud_data.csv` | 81134807 | 2026-09-11 10:36 |
| `data\raw\DATA_PROVENANCE.md` | 1323 | 2026-09-23 15:29 |
| `PROJECT_CONTINUATION_REPORT.md` | 2306 | 2026-10-09 16:27 |
| `pyproject.toml` | 371 | 2026-10-02 11:10 |
| `README.md` | 0 | 2026-09-09 02:34 |
| `reports\final_test_results.md` | 2024 | 2026-10-09 14:00 |
| `src\fraud_risk\__init__.py` | 0 | 2026-09-12 09:04 |
| `src\fraud_risk\config.py` | 144 | 2026-09-12 09:45 |
| `src\fraud_risk\data_loader.py` | 123 | 2026-09-12 09:50 |
| `src\fraud_risk\data_split.py` | 4704 | 2026-09-23 11:39 |
| `src\fraud_risk\data_validation.py` | 8365 | 2026-09-21 13:17 |
| `src\fraud_risk\eda.py` | 1232 | 2026-09-15 11:00 |
| `src\fraud_risk\evaluation.py` | 2337 | 2026-10-02 00:44 |
| `src\fraud_risk\model.py` | 1455 | 2026-10-02 11:21 |
| `src\fraud_risk\preprocessing.py` | 4681 | 2026-09-26 18:54 |
| `src\fraud_risk\run_evaluation.py` | 2542 | 2026-10-08 23:33 |
| `src\fraud_risk\run_final_evaluation.py` | 2443 | 2026-10-09 13:44 |
| `src\fraud_risk\selection_config.py` | 260 | 2026-10-09 12:38 |
| `src\fraud_risk\train_final_model.py` | 1510 | 2026-10-09 15:44 |
| `src\fraud_risk_platform.egg-info\dependency_links.txt` | 1 | 2026-10-08 23:24 |
| `src\fraud_risk_platform.egg-info\PKG-INFO` | 232 | 2026-10-08 23:24 |
| `src\fraud_risk_platform.egg-info\requires.txt` | 29 | 2026-10-08 23:24 |
| `src\fraud_risk_platform.egg-info\SOURCES.txt` | 543 | 2026-10-08 23:24 |
| `src\fraud_risk_platform.egg-info\top_level.txt` | 11 | 2026-10-08 23:24 |

## 3. Directory Structure

```text
fraud-risk-platform
  .gitignore
  data\raw\CCFraud_data.csv
  data\raw\DATA_PROVENANCE.md
  PROJECT_CONTINUATION_REPORT.md
  pyproject.toml
  README.md
  reports\final_test_results.md
  src\fraud_risk\__init__.py
  src\fraud_risk\config.py
  src\fraud_risk\data_loader.py
  src\fraud_risk\data_split.py
  src\fraud_risk\data_validation.py
  src\fraud_risk\eda.py
  src\fraud_risk\evaluation.py
  src\fraud_risk\model.py
  src\fraud_risk\preprocessing.py
  src\fraud_risk\run_evaluation.py
  src\fraud_risk\run_final_evaluation.py
  src\fraud_risk\selection_config.py
  src\fraud_risk\train_final_model.py
  src\fraud_risk_platform.egg-info\dependency_links.txt
  src\fraud_risk_platform.egg-info\PKG-INFO
  src\fraud_risk_platform.egg-info\requires.txt
  src\fraud_risk_platform.egg-info\SOURCES.txt
  src\fraud_risk_platform.egg-info\top_level.txt
```

## 4. Important Files and Tests

### Test-related files
- No obvious test files found.

### Configuration and dependency files
- `.gitignore`
- `pyproject.toml`

## 5. TODOs and Potentially Unfinished Work

These are search matches, not proof that a feature is incomplete.
- No matching markers found in scanned text files.

## 6. README and Dependency Manifests


### `pyproject.toml`
```text
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "fraud-risk-platform"
version = "0.1.0"
description = "Production-grade fraud risk machine learning platform"
requires-python = ">=3.13"
dependencies = [
    "pandas",
    "scikit-learn",
    "lightgbm",
]
[tool.setuptools.packages.find]
where = ["src"]
```

### `README.md`
```text

```

## 7. Source Code and Test Snapshot

> Files are included for review up to a combined 900 KB text budget.
> Oversized or binary files are listed in the inventory but not embedded.
> Secret-looking filenames are excluded; review this report before sharing it.

### `reports\final_test_results.md`
```
# Final Fraud Model Evaluation

**Evaluation date:** 2026-10-09  
**Model:** LightGBM  
**Decision threshold:** 0.65  
**Status:** Final test evaluation recorded

## 1. Evaluation setup

- Training data: months 1, 2, and 3.
- Reserved test data: month 7.
- Test transactions: 96,843.
- Test fraud rate: 1.4746%.
- Months 4–6 are absent from the available dataset.

The model and threshold were selected using development/validation data before evaluating the reserved test set.

## 2. Test results

| Metric | Result |
|---|---:|
| PR-AUC | 0.1976 |
| ROC-AUC | 0.8859 |
| Precision | 0.1527 |
| Recall | 0.5105 |
| Threshold | 0.65 |

## 3. Confusion matrix

| Outcome | Count |
|---|---:|
| True negatives | 91,371 |
| False positives | 4,044 |
| False negatives | 699 |
| True positives | 729 |
| Total alerts | 4,773 |

## 4. Interpretation

At the selected threshold, the model detected 729 fraudulent transactions and missed 699. It generated 4,044 false-positive alerts.

Test recall (51.05%) was lower than validation recall (70.55%). This indicates that the selected operating point did not generalize as well to the reserved test month.

The test fraud prevalence (1.4746%) was higher than validation prevalence (0.9222%). This affects how precision should be compared between the two periods.

## 5. Limitations and next steps

- Months 4–6 are absent, so performance across those periods is unknown.
- The test results do not establish that the model is production-ready.
- The 0.65 threshold is a provisional business decision. False-positive and false-negative costs have not been validated with business stakeholders.
- Do not tune the model or threshold against this reserved test set. Use development data for further experiments and retain these results as the evaluation of the current selection procedure.

Next engineering steps: package the fitted pipeline, add automated tests, expose predictions through an API, containerize the service, configure CI, and document deployment and monitoring.

```

### `src\fraud_risk\model.py`
```
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
```

### `src\fraud_risk\train_final_model.py`
```
"""Train and package the selected fraud detection model."""

import joblib
import pandas as pd
from pathlib import Path

from fraud_risk.data_loader import load_data
from fraud_risk.data_split import split_data
from fraud_risk.model import build_lightgbm_model
from fraud_risk.selection_config import SELECTED_MODEL
from fraud_risk.preprocessing import TARGET_COLUMN, TIME_COLUMN

# --- Artifact configuration ---------------------------------------------

MODEL_DIR = Path("artifacts")
MODEL_PATH = MODEL_DIR / "fraud_model.joblib"

# --- Training ------------------------------------------------------------

def train_final_model():
"""Fit the selected pipeline on training and validation data."""

```
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
```

# --- Artifact persistence -----------------------------------------------

def main():
MODEL_DIR.mkdir(parents=True, exist_ok=True)

```
model = train_final_model()
joblib.dump(model, MODEL_PATH)

print(f"Selected model: {SELECTED_MODEL}")
print(f"Artifact saved: {MODEL_PATH}")
```

if **name** == "**main**":
main()

```

### `src\fraud_risk\config.py`
```
from pathlib import Path

PROJECT_NAME = "fraud-risk-platform"
DATA_DIR = Path("data") / "raw"
RAW_DATA_FILE = DATA_DIR / "CCFraud_data.csv"
```

### `src\fraud_risk\data_loader.py`
```
import pandas as pd
from fraud_risk.config import RAW_DATA_FILE

def load_data():
    return pd.read_csv(RAW_DATA_FILE)
```

### `src\fraud_risk\data_split.py`
```

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

```

### `src\fraud_risk\data_validation.py`
```
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
```

### `src\fraud_risk\eda.py`
```
from fraud_risk.data_loader import load_data


def main():
    df = load_data()

    fraud = df[df["fraud_bool"] == 1]
    non_fraud = df[df["fraud_bool"] == 0]

    print("Fraud cases:", len(fraud))
    print("Non-fraud cases:", len(non_fraud))
    print()

    print("Fraud rate:", len(fraud) / len(df) * 100)
    print()

    print("Fraud group:")
    print(fraud.describe(include="all").T)

    print()
    print("Non-fraud group:")
    print(non_fraud.describe(include="all").T)

    print(df["date_of_birth_distinct_emails_4w"].describe())
    print()
    print(df["date_of_birth_distinct_emails_4w"].value_counts().sort_index().head(20))

    print("\nDate-of-birth distinct emails range:")
    print("Minimum:", df["date_of_birth_distinct_emails_4w"].min())
    print("Maximum:", df["date_of_birth_distinct_emails_4w"].max())
    print("Unique values:", df["date_of_birth_distinct_emails_4w"].nunique())

    print("\nDOB distinct emails:")
    print("Minimum:", df["date_of_birth_distinct_emails_4w"].min())
    print("Maximum:", df["date_of_birth_distinct_emails_4w"].max())
    print("Unique:", df["date_of_birth_distinct_emails_4w"].nunique())

if __name__ == "__main__":
    main()
```

### `src\fraud_risk\evaluation.py`
```
"""
Evaluation utilities for the fraud detection model.

Provides metrics for measuring model performance on unseen data.
"""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X, y, threshold=0.5):
    """Calculate classification metrics for a fitted fraud model."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predictions),
    }



# --- Threshold configuration --------------------------------------------

DEFAULT_THRESHOLDS = tuple(
    round(index / 100, 2)
    for index in range(1, 100)
)

# --- Threshold evaluation ------------------------------------------------

def evaluate_thresholds(model, X, y, thresholds, fp_cost=1.0, fn_cost=1.0):
    """Evaluate model performance and error cost across thresholds."""
    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()

        total_cost = (fp * fp_cost) + (fn * fn_cost)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "alerts": int(fp + tp),
                "total_cost": float(total_cost),
            }
        )

    return results
```

### `src\fraud_risk\preprocessing.py`
```
"""
Preprocessing pipeline for the fraud detection model.

Builds a leak-safe ColumnTransformer that:
- converts sentinel (-1) values to NaN in selected numeric columns
- imputes missing numeric/categorical values (median / most frequent)
- adds missingness indicators for sentinel-coded features
- scales numeric features and one-hot encodes categoricals

All statistics (medians, categories) are learned from training data only,
via build_preprocessor().fit(X_train).
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --- Feature contract -------------------------------------------------

TARGET_COLUMN = "fraud_bool"
TIME_COLUMN = "month"  # used for chronological train/val/test splits

CATEGORICAL_FEATURES = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]

# -1 is a sentinel for "missing" in these columns; must be converted
# to NaN before imputation, unlike ordinary numeric features below.
SENTINEL_NUMERIC_FEATURES = [
    "prev_address_months_count",
    "bank_months_count",
    "current_address_months_count",
    "session_length_in_minutes",
    "device_distinct_emails_8w",
]

NUMERIC_FEATURES = [
    "income",
    "name_email_similarity",
    "customer_age",
    "days_since_request",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_6h",
    "velocity_24h",
    "velocity_4w",
    "bank_branch_count_8w",
    "date_of_birth_distinct_emails_4w",
    "credit_risk_score",
    "email_is_free",
    "phone_home_valid",
    "phone_mobile_valid",
    "has_other_cards",
    "proposed_credit_limit",
    "foreign_request",
    "keep_alive_session",
    "device_fraud_count",
]


class SentinelToNaN(BaseEstimator, TransformerMixin):
    """Replace a sentinel value (default -1) with NaN."""

    def __init__(self, sentinel_value=-1):
        self.sentinel_value = sentinel_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Copy to avoid mutating the caller's DataFrame.
        return X.copy().replace(self.sentinel_value, float("nan"))

    def get_feature_names_out(self, input_features=None):
        """Preserve the original feature names after transformation."""
        return input_features


# --- Per-feature-group pipelines ---------------------------------------

# Sentinel numerics: unmask -1 as missing, impute median, keep an
# indicator flag (missingness itself can be predictive), then scale.
sentinel_numeric_pipeline = Pipeline([
    ("sentinel_to_nan", SentinelToNaN(sentinel_value=-1)),
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])

# Regular numerics: no sentinel handling needed.
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categoricals: fill with most frequent, one-hot encode.
# handle_unknown="ignore" avoids failures on unseen categories at inference.
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
])


# --- Combined preprocessor ----------------------------------------------

# remainder="drop" is the leakage guard: TARGET_COLUMN and TIME_COLUMN
# are never listed below, so they can't accidentally reach the model.
preprocessor = ColumnTransformer(
    transformers=[
        ("sentinel_numeric", sentinel_numeric_pipeline, SENTINEL_NUMERIC_FEATURES),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)


def build_preprocessor():
    """Return a fresh, unfitted preprocessing transformer.

    Call .fit() only on training data to avoid leaking validation/test
    statistics into the pipeline.
    """
    return ColumnTransformer(
        transformers=[
            (
                "sentinel_numeric",
                sentinel_numeric_pipeline,
                SENTINEL_NUMERIC_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )
```

### `src\fraud_risk\run_evaluation.py`
```

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
```

### `src\fraud_risk\run_final_evaluation.py`
```

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
    """Train on months 1â€“3 and evaluate once on month 7."""
    df = load_data()
    train_df, validation_df, test_df = split_data(df)

    # Confirm the reserved test partition matches our configuration.
    actual_test_months = set(test_df["month"].unique())
    if actual_test_months != {TEST_MONTH}:
        raise ValueError(
            f"Expected test month {TEST_MONTH}, "
            f"found {sorted(actual_test_months)}"
        )

    # After model and threshold selection, combine months 1â€“3
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
```

### `src\fraud_risk\selection_config.py`
```
"""Locked model-selection criteria."""

# Selected using validation data only.
SELECTED_MODEL = "lightgbm"
SELECTED_THRESHOLD = 0.65

# Minimum acceptable validation recall.
MINIMUM_RECALL = 0.70

# Month reserved for final evaluation.
TEST_MONTH = 7
```

### `data\raw\DATA_PROVENANCE.md`
```
# Dataset Provenance

## Dataset
- **File:** `CCFraud_data.csv`
- **Rows:** 500,000
- **Columns:** 32

## Provenance
A 500,000-row republished copy/subset of the Bank Account Fraud (BAF) dataset family, obtained from a third-party GitHub repository.

The exact upstream source and transformation used to produce this 500,000-row copy cannot currently be established from the GitHub repository alone.

The dataset schema matches the BAF dataset family.

- **Date obtained:** September 11, 2026
- **Source URL:** Not recorded at time of download; could not be recovered from browser history as of September 23, 2026.

## License & Redistribution
Redistribution rights for this specific copy are unverified. The raw CSV is excluded from version control accordingly (see `data/raw/*.csv` in `.gitignore`).

## Integrity Verification

**SHA-256:**
```
16CD960780E25F88682CA86ACC56BE5016360B5468A60E3DD8F68F415716B77D
```

To verify the file has not changed:

```powershell
Get-FileHash -Algorithm SHA256 "data\raw\CCFraud_data.csv"
```

```bash
shasum -a 256 data/raw/CCFraud_data.csv
```

## Important Note
This dataset is synthetic/privacy-preserving data derived from an anonymized real-world bank-account fraud dataset. It should not be described as raw real-world banking transactions.
```

### `PROJECT_CONTINUATION_REPORT.md`
```
# PROJECT CONTINUATION REPORT
Generated: 2026-10-09 16:27:04 +01:00
Project folder: C:\Users\HP\Documents\#ML_ENGINE\fraud-risk-platform
Machine: DESKTOP-L1060FJ
Report purpose: Preserve project state for an accurate development handoff.

IMPORTANT:
- This report is a snapshot, not proof that the application works.
- Tests are not automatically executed by this script.
- Secret files and common virtual-environment folders are excluded.
- Review this report for private information before sharing it.

## GIT STATUS

## main...origin/main
?? PROJECT_CONTINUATION_REPORT.md
?? src/fraud_risk/run_final_evaluation.py
?? src/fraud_risk/train_final_model.py


## CURRENT BRANCH

main


## LATEST 15 COMMITS

3a4adee | 2026-10-09T14:04:53+01:00 | Add final fraud model evaluation report
e1bc7b9 | 2026-10-09T13:40:28+01:00 | Add fraud model evaluation and selection baseline
eb74594 | 2026-09-26T18:57:32+01:00 | Add leakage-safe preprocessing pipeline
e5fa999 | 2026-09-23T15:29:30+01:00 | Add date obtained, redistribution note, and hash verification steps to data provenance
d38e973 | 2026-09-23T12:58:26+01:00 | Add chronological data split with validation checks
7f353fc | 2026-09-09T05:37:59+01:00 | Initial project structure


## TRACKED CHANGE SUMMARY




## STAGED CHANGE SUMMARY




## UNTRACKED FILES

PROJECT_CONTINUATION_REPORT.md
src/fraud_risk/run_final_evaluation.py
src/fraud_risk/train_final_model.py


## TRACKED FILES

.gitignore
README.md
data/raw/DATA_PROVENANCE.md
pyproject.toml
reports/final_test_results.md
src/fraud_risk/__init__.py
src/fraud_risk/config.py
src/fraud_risk/data_loader.py
src/fraud_risk/data_split.py
src/fraud_risk/data_validation.py
src/fraud_risk/eda.py
src/fraud_risk/evaluation.py
src/fraud_risk/model.py
src/fraud_risk/preprocessing.py
src/fraud_risk/run_evaluation.py
src/fraud_risk/selection_config.py


## GIT REMOTES (REDACTED)

origin	https://github.com/increase-codes/fraud-risk-platform.git (fetch)
origin	https://github.com/increase-codes/fraud-risk-platform.git (push)


## PYTHON VERSION

Python 3.13.5



## PYTHON EXECUTABLE

C:\Users\HP\AppData\Local\Programs\Python\Python313\python.exe


## VIRTUAL ENVIRONMENT

VIRTUAL_ENV = Not activated or not detected


```

### `pyproject.toml`
```
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "fraud-risk-platform"
version = "0.1.0"
description = "Production-grade fraud risk machine learning platform"
requires-python = ">=3.13"
dependencies = [
    "pandas",
    "scikit-learn",
    "lightgbm",
]
[tool.setuptools.packages.find]
where = ["src"]
```

### `src\fraud_risk_platform.egg-info\dependency_links.txt`
```


```

### `src\fraud_risk_platform.egg-info\requires.txt`
```
pandas
scikit-learn
lightgbm

```

### `src\fraud_risk_platform.egg-info\SOURCES.txt`
```
README.md
pyproject.toml
src/fraud_risk/__init__.py
src/fraud_risk/config.py
src/fraud_risk/data_loader.py
src/fraud_risk/data_split.py
src/fraud_risk/data_validation.py
src/fraud_risk/eda.py
src/fraud_risk/evaluation.py
src/fraud_risk/model.py
src/fraud_risk/preprocessing.py
src/fraud_risk/run_evaluation.py
src/fraud_risk_platform.egg-info/PKG-INFO
src/fraud_risk_platform.egg-info/SOURCES.txt
src/fraud_risk_platform.egg-info/dependency_links.txt
src/fraud_risk_platform.egg-info/requires.txt
src/fraud_risk_platform.egg-info/top_level.txt
```

### `src\fraud_risk_platform.egg-info\top_level.txt`
```
fraud_risk

```

## 8. Next Audit Checklist

- [ ] Confirm the current Git branch and whether local changes are committed.
- [ ] Read the project README and establish the intended architecture.
- [ ] Identify the application entry point and end-to-end data flow.
- [ ] Check model training, preprocessing, leakage risks, and evaluation.
- [ ] Inspect tests and identify which tests actually pass.
- [ ] Check API, deployment, monitoring, and reproducibility requirements.
- [ ] Compare existing features with the original project goals.
- [ ] Create a prioritized list of remaining work.

## 9. Report Limitations

- This script does not execute the application or tests.
- Git remotes do not prove that remote code matches the local working tree.
- Ignored directories, sensitive files, binary datasets, and model artifacts are excluded.
- Source contents may be truncated; inspect the inventory for omissions.
- No feature should be declared complete solely because a file exists.

End of report.
