import numpy as np
import pytest

from fraud_risk.drift_detection import (
    calculate_psi,
    detect_numeric_drift,
)


def test_identical_data_has_low_psi():
    values = np.random.default_rng(1).normal(size=2000)

    assert calculate_psi(values, values) < 0.01


def test_shifted_data_detects_drift():
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, 5000)
    current = rng.normal(2, 1, 5000)

    results = detect_numeric_drift(
        __import__("pandas").DataFrame({"amount": reference}),
        __import__("pandas").DataFrame({"amount": current}),
    )

    assert results["amount"]["drift_detected"] is True
    assert results["amount"]["psi"] >= 0.2


def test_missing_reference_column_raises_error():
    with pytest.raises(ValueError, match="missing columns"):
        detect_numeric_drift(
            __import__("pandas").DataFrame({"amount": [1, 2, 3]}),
            __import__("pandas").DataFrame({"other": [1, 2, 3]}),
        )
