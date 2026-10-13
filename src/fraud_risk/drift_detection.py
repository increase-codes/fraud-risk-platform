
"""Detect numeric feature drift using Population Stability Index (PSI)."""

import numpy as np
import pandas as pd


def calculate_psi(reference, current, bins=10):
    """Calculate PSI for one numeric feature."""
    reference = pd.to_numeric(
        pd.Series(reference), errors="coerce"
    ).dropna()
    current = pd.to_numeric(
        pd.Series(current), errors="coerce"
    ).dropna()

    if reference.empty or current.empty:
        raise ValueError("Reference and current data must contain numeric values.")

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = ref_counts / len(reference)
    cur_pct = cur_counts / len(current)

    epsilon = 1e-6
    ref_pct = np.clip(ref_pct, epsilon, None)
    cur_pct = np.clip(cur_pct, epsilon, None)

    return float(np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    ))


def detect_numeric_drift(reference, current, threshold=0.2):
    """Report PSI and flag features exceeding the threshold."""
    if not 0 < threshold:
        raise ValueError("Threshold must be positive.")

    missing = sorted(set(reference.columns) - set(current.columns))
    if missing:
        raise ValueError(f"Current data is missing columns: {missing}")

    results = {}

    for column in reference.select_dtypes(include="number").columns:
        if column not in current.columns:
            raise ValueError(f"Current data is missing column: {column}")

        psi = calculate_psi(reference[column], current[column])
        results[column] = {
            "psi": round(psi, 6),
            "drift_detected": psi >= threshold,
        }

    return results