"""
Evaluation utilities for the fraud detection model.

Provides metrics for measuring model performance on unseen data.
"""

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X, y, threshold=0.5):
    """Calculate classification metrics for a fitted fraud model."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc": average_precision_score(y, probabilities),
        "roc_auc": roc_auc_score(y, probabilities),
        "precision": precision_score(y, predictions, zero_division=0),
        "recall": recall_score(y, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predictions),
    }



# --- Threshold configuration --------------------------------------------

DEFAULT_THRESHOLDS = tuple(
    round(index / 100, 2)
    for index in range(1, 100)
)

# --- Threshold evaluation ------------------------------------------------

def evaluate_thresholds(model, X, y, thresholds, fp_cost=1.0, fn_cost=1.0):
    """Evaluate model performance and error cost across thresholds."""
    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).ravel()

        total_cost = (fp * fp_cost) + (fn * fn_cost)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y,
                    predictions,
                    zero_division=0,
                ),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "alerts": int(fp + tp),
                "total_cost": float(total_cost),
            }
        )

    return results