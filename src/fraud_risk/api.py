"""FastAPI service for fraud predictions."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from fraud_risk.predict import load_model, predict_fraud

app = FastAPI(title="Fraud Risk API", version="1.0.0")


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(transaction: Transaction):
    try:
        import pandas as pd

        data = pd.DataFrame([transaction.features])
        model = load_model()
        result = predict_fraud(data, model)

        return {
            "fraud_probability": float(result.loc[0, "fraud_probability"]),
            "fraud_prediction": int(result.loc[0, "fraud_prediction"]),
        }
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
