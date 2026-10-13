
import json

import pandas as pd
from fastapi.testclient import TestClient

import fraud_risk.api as api
from fraud_risk.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SENTINEL_NUMERIC_FEATURES,
)

client = TestClient(api.app)


def valid_features():
    features = {
        name: 1.0
        for name in NUMERIC_FEATURES + SENTINEL_NUMERIC_FEATURES
    }
    features.update({
        name: "test_value"
        for name in CATEGORICAL_FEATURES
    })
    return features


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_accepts_valid_transaction(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: object())
    monkeypatch.setattr(
        api,
        "predict_fraud",
        lambda data, model: pd.DataFrame({
            "fraud_probability": [0.25],
            "fraud_prediction": [0],
        }),
    )

    response = client.post(
        "/predict",
        json={"features": valid_features()},
    )

    assert response.status_code == 200
    assert response.json() == {
        "fraud_probability": 0.25,
        "fraud_prediction": 0,
    }


def test_predict_rejects_missing_features():
    response = client.post(
        "/predict",
        json={"features": {"income": 50000}},
    )

    assert response.status_code == 422
    assert "payment_type" in response.json()["detail"]["missing_features"]


def test_predict_rejects_unexpected_features():
    features = valid_features()
    features["fake_feature"] = 123

    response = client.post(
        "/predict",
        json={"features": features},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["unexpected_features"] == [
        "fake_feature"
    ]


def test_predict_returns_503_when_model_is_unavailable(monkeypatch):
    def fail_to_load_model():
        raise FileNotFoundError("Model artifact not found")

    monkeypatch.setattr(api, "load_model", fail_to_load_model)

    response = client.post(
        "/predict",
        json={"features": valid_features()},
    )

    assert response.status_code == 503


def test_predict_handles_unexpected_prediction_failure(monkeypatch):
    monkeypatch.setattr(api, "load_model", lambda: object())

    def fail_prediction(data, model):
        raise RuntimeError("Unexpected model failure")

    monkeypatch.setattr(api, "predict_fraud", fail_prediction)

    response = client.post(
        "/predict",
        json={"features": valid_features()},
    )

    assert response.status_code == 500


def test_metrics_endpoint(monkeypatch, tmp_path):
    report_path = tmp_path / "drift_report.json"
    report_path.write_text(
        json.dumps({
            "generated_at_utc": "2026-10-13T00:00:00+00:00",
            "threshold": 0.2,
            "partitions": {
                "validation_month_3": {
                    "features_checked": 25,
                    "drifted_features": 3,
                    "drift_detected": True,
                }
            },
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(api, "DRIFT_REPORT_FILE", report_path)

    response = client.get("/metrics")

    assert response.status_code == 200
    metrics = response.json()
    assert "prediction_requests" in metrics
    assert "successful_predictions" in metrics
    assert "failed_predictions" in metrics
    assert "average_latency_ms" in metrics
    assert "max_latency_ms" in metrics

    drift = metrics["drift_report"]
    assert drift["status"] == "available"
    assert drift["partitions"]["validation_month_3"]["drift_detected"] is True
    assert drift["partitions"]["validation_month_3"]["drifted_features"] == 3


def test_metrics_reports_missing_drift_report(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api, "DRIFT_REPORT_FILE", tmp_path / "missing.json"
    )

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["drift_report"] == {
        "status": "not_available"
    }


def test_metrics_reports_invalid_drift_report(monkeypatch, tmp_path):
    report_path = tmp_path / "invalid.json"
    report_path.write_text("not valid JSON", encoding="utf-8")
    monkeypatch.setattr(api, "DRIFT_REPORT_FILE", report_path)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.json()["drift_report"] == {"status": "invalid"}
