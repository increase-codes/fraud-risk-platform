import json

import numpy as np
import pandas as pd

import fraud_risk.run_drift_check as drift_report


def test_drift_report_is_saved(monkeypatch, tmp_path):
    rng = np.random.default_rng(42)

    train = pd.DataFrame({
        "month": [1, 2] * 500,
        "fraud_bool": [0, 1] * 500,
        "amount": rng.normal(0, 1, 1000),
    })
    validation = pd.DataFrame({
        "month": [3] * 1000,
        "fraud_bool": [0] * 1000,
        "amount": rng.normal(3, 1, 1000),
    })
    test = pd.DataFrame({
        "month": [7] * 1000,
        "fraud_bool": [0] * 1000,
        "amount": rng.normal(4, 1, 1000),
    })

    monkeypatch.setattr(
        drift_report, "load_data", lambda: pd.DataFrame()
    )
    monkeypatch.setattr(
        drift_report,
        "split_data",
        lambda df: (train, validation, test),
    )

    report_path = tmp_path / "drift_report.json"
    monkeypatch.setattr(drift_report, "REPORT_FILE", report_path)

    drift_report.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["reference_months"] == [1, 2]
    assert report["partitions"]["validation_month_3"]["drift_detected"]
    assert report["partitions"]["test_month_7"]["drift_detected"]
    assert "fraud_bool" not in report["partitions"]["test_month_7"]["features"]