
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