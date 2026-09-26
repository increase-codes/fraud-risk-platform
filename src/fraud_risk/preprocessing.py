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