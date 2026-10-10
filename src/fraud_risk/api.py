"""FastAPI service for fraud predictions."""

import math

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from fraud_risk.predict import load_model, predict_fraud
from fraud_risk.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SENTINEL_NUMERIC_FEATURES,
)

app = FastAPI(title="Fraud Risk API", version="1.0.0")

NUMERIC_INPUTS = set(NUMERIC_FEATURES + SENTINEL_NUMERIC_FEATURES)
CATEGORICAL_INPUTS = set(CATEGORICAL_FEATURES)
REQUIRED_FEATURES = NUMERIC_INPUTS | CATEGORICAL_INPUTS


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    features = transaction.features
    supplied = set(features)

    missing = sorted(REQUIRED_FEATURES - supplied)
    unexpected = sorted(supplied - REQUIRED_FEATURES)

    if missing or unexpected:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": missing,
                "unexpected_features": unexpected,
            },
        )

    for name in NUMERIC_INPUTS:
        value = features[name]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be numeric.",
            )
        if not math.isfinite(value):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be finite.",
            )

    for name in CATEGORICAL_INPUTS:
        if not isinstance(features[name], str):
            raise HTTPException(
                status_code=422,
                detail=f"Feature '{name}' must be a string.",
            )

    try:
        model = load_model()
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable. Train the model first.",
        ) from exc

    data = pd.DataFrame([features])

    try:
        result = predict_fraud(data, model)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "fraud_probability": float(result.loc[0, "fraud_probability"]),
        "fraud_prediction": int(result.loc[0, "fraud_prediction"]),
    }
