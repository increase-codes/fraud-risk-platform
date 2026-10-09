"""Locked model-selection criteria."""

# Selected using validation data only.
SELECTED_MODEL = "lightgbm"
SELECTED_THRESHOLD = 0.65

# Minimum acceptable validation recall.
MINIMUM_RECALL = 0.70

# Month reserved for final evaluation.
TEST_MONTH = 7