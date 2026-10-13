
"""FastAPI service for fraud predictions and monitoring."""

import json
import logging
import math
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from fraud_risk.monitoring import (
    get_metrics as get_monitoring_metrics,
    record_prediction,
)
from fraud_risk.predict import load_model, predict_fraud
from fraud_risk.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SENTINEL_NUMERIC_FEATURES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_risk.api")

app = FastAPI(title="Fraud Risk API", version="1.0.0")

DRIFT_REPORT_FILE = Path("reports") / "drift_report.json"

NUMERIC_INPUTS = set(NUMERIC_FEATURES + SENTINEL_NUMERIC_FEATURES)
CATEGORICAL_INPUTS = set(CATEGORICAL_FEATURES)
REQUIRED_FEATURES = NUMERIC_INPUTS | CATEGORICAL_INPUTS


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | str]


def get_drift_report_status() -> dict:
    """Return a compact summary of the latest saved drift report."""
    try:
        report = json.loads(
            DRIFT_REPORT_FILE.read_text(encoding="utf-8")
        )
    except FileNotFoundError:
        return {"status": "not_available"}
    except (OSError, json.JSONDecodeError):
        return {"status": "invalid"}

    if not isinstance(report, dict):
        return {"status": "invalid"}

    partitions = report.get("partitions")
    if not isinstance(partitions, dict):
        return {"status": "invalid"}

    summary = {}
    for name, details in partitions.items():
        if not isinstance(details, dict):
            return {"status": "invalid"}

        summary[name] = {
            "features_checked": details.get("features_checked"),
            "drifted_features": details.get("drifted_features"),
            "drift_detected": details.get("drift_detected"),
        }

    return {
        "status": "available",
        "generated_at_utc": report.get("generated_at_utc"),
        "threshold": report.get("threshold"),
        "partitions": summary,
    }


@app.middleware("http")
async def monitor_prediction_requests(request: Request, call_next):
    if request.method != "POST" or request.url.path != "/predict":
        return await call_next(request)

    start_time = time.perf_counter()
    success = False

    try:
        response = await call_next(request)
        success = response.status_code < 400
        return response
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        record_prediction(success, latency_ms)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    result = get_monitoring_metrics()
    result["drift_report"] = get_drift_report_status()
    return result


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
        logger.exception("model_artifact_unavailable")
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable. Train the model first.",
        ) from exc

    data = pd.DataFrame([features])

    try:
        result = predict_fraud(data, model)
    except (ValueError, KeyError) as exc:
        logger.warning("prediction_failed: %s", str(exc))
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.exception("unexpected_prediction_failure")
        raise HTTPException(
            status_code=500,
            detail="Internal prediction error.",
        ) from None

    fraud_probability = float(result.loc[0, "fraud_probability"])
    fraud_prediction = int(result.loc[0, "fraud_prediction"])

    logger.info(
        "prediction_completed fraud_probability=%.6f "
        "fraud_prediction=%d",
        fraud_probability,
        fraud_prediction,
    )

    return {
        "fraud_probability": fraud_probability,
        "fraud_prediction": fraud_prediction,
    }
