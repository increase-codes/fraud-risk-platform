# Final Fraud Model Evaluation

**Evaluation date:** 2026-10-09  
**Model:** LightGBM  
**Decision threshold:** 0.65  
**Status:** Final test evaluation recorded

## 1. Evaluation setup

- Training data: months 1, 2, and 3.
- Reserved test data: month 7.
- Test transactions: 96,843.
- Test fraud rate: 1.4746%.
- Months 4–6 are absent from the available dataset.

The model and threshold were selected using development/validation data before evaluating the reserved test set.

## 2. Test results

| Metric | Result |
|---|---:|
| PR-AUC | 0.1976 |
| ROC-AUC | 0.8859 |
| Precision | 0.1527 |
| Recall | 0.5105 |
| Threshold | 0.65 |

## 3. Confusion matrix

| Outcome | Count |
|---|---:|
| True negatives | 91,371 |
| False positives | 4,044 |
| False negatives | 699 |
| True positives | 729 |
| Total alerts | 4,773 |

## 4. Interpretation

At the selected threshold, the model detected 729 fraudulent transactions and missed 699. It generated 4,044 false-positive alerts.

Test recall (51.05%) was lower than validation recall (70.55%). This indicates that the selected operating point did not generalize as well to the reserved test month.

The test fraud prevalence (1.4746%) was higher than validation prevalence (0.9222%). This affects how precision should be compared between the two periods.

## 5. Limitations and next steps

- Months 4–6 are absent, so performance across those periods is unknown.
- The test results do not establish that the model is production-ready.
- The 0.65 threshold is a provisional business decision. False-positive and false-negative costs have not been validated with business stakeholders.
- Do not tune the model or threshold against this reserved test set. Use development data for further experiments and retain these results as the evaluation of the current selection procedure.

Next engineering steps: package the fitted pipeline, add automated tests, expose predictions through an API, containerize the service, configure CI, and document deployment and monitoring.
