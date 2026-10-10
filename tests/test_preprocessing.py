import pandas as pd

from fraud_risk.preprocessing import SentinelToNaN


def test_sentinel_values_are_replaced_with_nan():
    df = pd.DataFrame({
        "prev_address_months_count": [-1, 5, 12],
        "bank_months_count": [3, -1, 8],
    })

    transformed = SentinelToNaN().transform(df)

    assert pd.isna(transformed.loc[0, "prev_address_months_count"])
    assert pd.isna(transformed.loc[1, "bank_months_count"])
    assert transformed.loc[1, "prev_address_months_count"] == 5
    assert transformed.loc[0, "bank_months_count"] == 3


def test_sentinel_transform_does_not_mutate_original_data():
    df = pd.DataFrame({
        "prev_address_months_count": [-1, 5],
    })

    SentinelToNaN().transform(df)

    assert df.loc[0, "prev_address_months_count"] == -1
