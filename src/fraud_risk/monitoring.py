
"""Simple in-process monitoring for fraud predictions."""

from threading import Lock

_lock = Lock()

_metrics = {
    "prediction_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_latency_ms": 0.0,
    "max_latency_ms": 0.0,
}


def record_prediction(success: bool, latency_ms: float) -> None:
    """Record the outcome and latency of a prediction attempt."""
    with _lock:
        _metrics["prediction_requests"] += 1

        if success:
            _metrics["successful_predictions"] += 1
        else:
            _metrics["failed_predictions"] += 1

        _metrics["total_latency_ms"] += latency_ms
        _metrics["max_latency_ms"] = max(
            _metrics["max_latency_ms"], latency_ms
        )


def get_metrics() -> dict[str, int | float]:
    """Return a safe snapshot of the current metrics."""
    with _lock:
        result = _metrics.copy()

    requests = result["prediction_requests"]
    result["average_latency_ms"] = (
        round(result["total_latency_ms"] / requests, 2)
        if requests
        else 0.0
    )
    result["total_latency_ms"] = round(result["total_latency_ms"], 2)

    return result