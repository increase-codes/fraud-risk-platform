
import pandas as pd
import pytest

from fraud_risk.data_split import split_data


def sample_data():
    return pd.DataFrame({
        "month": [1, 2, 3, 7, 1, 3, 7],
        "fraud_bool": [0, 1, 0, 1, 0, 0, 0],
    })


def test_split_uses_expected_months():
    train, validation, test = split_data(sample_data())

    assert set(train["month"]) == {1, 2}
    assert set(validation["month"]) == {3}
    assert set(test["month"]) == {7}


def test_split_has_no_overlapping_rows():
    train, validation, test = split_data(sample_data())

    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)


def test_split_rejects_unexpected_month():
    df = sample_data()
    df.loc[0, "month"] = 4

    with pytest.raises(ValueError, match="Unexpected month"):
        split_data(df)


def test_split_rejects_empty_training_partition():
    df = sample_data()
    df = df[~df["month"].isin([1, 2])]

    with pytest.raises(ValueError, match="Training dataset is empty"):
        split_data(df)

def test_split_rejects_missing_expected_month():
    df = sample_data()
    df = df[df["month"] != 2]

    with pytest.raises(ValueError, match="Missing expected month"):
        split_data(df)