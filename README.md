# Fraud Risk Platform

A Python machine-learning engineering project for detecting potentially fraudulent transactions using LightGBM. The platform includes data validation, chronological evaluation, a reusable prediction pipeline, a FastAPI service, Docker packaging, automated tests, prediction monitoring, offline drift detection, and a guarded candidate-retraining workflow.

**Live deployment:** [https://fraud-risk-platform-2pb9.onrender.com](https://fraud-risk-platform-2pb9.onrender.com)

**Interactive API documentation:** [https://fraud-risk-platform-2pb9.onrender.com/docs](https://fraud-risk-platform-2pb9.onrender.com/docs)

**Repository:** [https://github.com/increase-codes/fraud-risk-platform](https://github.com/increase-codes/fraud-risk-platform)

## Project status

The project demonstrates an end-to-end ML engineering workflow, from data preparation and model evaluation to API serving, containerization, CI, basic monitoring, drift analysis, and candidate-model validation.

The application has been deployed to Render and its health and prediction endpoints have previously been exercised. The automated test suite has also passed locally and through GitHub Actions.

**This is an engineering showcase, not a claim of production readiness.** Persistent monitoring, live drift calculation, alerting, operational retraining orchestration, stronger deployment safeguards, and further model validation remain areas for improvement.

The recorded test metrics also show that the current model misses a substantial proportion of fraudulent transactions. Any real-world use would require additional validation and an appropriate human-review and risk-management process.

## Features

- Data loading, validation, and exploratory analysis.
- Chronological training, validation, and test partitions.
- Numeric and categorical feature preprocessing.
- Logistic Regression baseline and LightGBM classification.
- Model serialization and final evaluation.
- Configurable prediction threshold.
- FastAPI prediction service with input validation and error handling.
- Docker-based application packaging.
- GitHub Actions continuous integration.
- Prediction request, success, failure, and latency metrics.
- Population Stability Index (PSI) drift analysis and JSON reporting.
- Candidate retraining with a validation-recall acceptance gate.
- Automated tests for data splitting, preprocessing, API behavior, drift detection, reporting, and retraining-gate logic.

## Technology stack

- Python 3.13+
- pandas
- scikit-learn
- LightGBM
- joblib
- FastAPI
- Uvicorn
- pytest
- Docker
- GitHub Actions
- Render

## Repository structure

The following tree shows the principal project files. The dataset is expected locally but is not distributed with the repository. Some generated reports and the model artifact may already exist in the working copy.

```text
fraud-risk-platform/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/
│       └── DATA_PROVENANCE.md
├── reports/
│   ├── final_test_results.md
│   └── drift_report.json
├── artifacts/
│   └── fraud_model.joblib
├── src/
│   └── fraud_risk/
│       ├── api.py
│       ├── config.py
│       ├── data_loader.py
│       ├── data_split.py
│       ├── data_validation.py
│       ├── drift_detection.py
│       ├── eda.py
│       ├── evaluation.py
│       ├── model.py
│       ├── monitoring.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── retraining.py
│       ├── run_drift_check.py
│       ├── run_evaluation.py
│       ├── run_final_evaluation.py
│       ├── selection_config.py
│       └── train_final_model.py
├── tests/
│   ├── test_api.py
│   ├── test_data_split.py
│   ├── test_drift_detection.py
│   ├── test_drift_report.py
│   ├── test_preprocessing.py
│   └── test_retraining.py
├── Dockerfile
├── pyproject.toml
└── README.md
```

The model artifact `artifacts/fraud_model.joblib` is tracked by Git in the current repository to support the existing deployment. Treat serialized model files as versioned release artifacts and verify their provenance and compatibility when changing the model or dependencies.

## Installation

Requires Python 3.13 or later.

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/increase-codes/fraud-risk-platform.git
cd fraud-risk-platform
```

Create and activate a virtual environment:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest httpx
```

If Python 3.13 is already available through `python`, you can use `python -m venv .venv` instead.

## Dataset setup

The training and evaluation workflows expect the dataset at:

```text
data/raw/CCFraud_data.csv
```

The dataset is not distributed with this repository because redistribution permissions have not been verified. Obtain it from an authorized source and review `data/raw/DATA_PROVENANCE.md` before use.

The dataset must contain the target column `fraud_bool`, the temporal column `month`, and the feature columns expected by the project's data-processing and modeling modules.

### Chronological data split

The project uses the following partitions:

- **Training:** months 1 and 2.
- **Validation:** month 3.
- **Test:** month 7.

The test partition is reserved for final evaluation. It should not be used to select models, tune thresholds, or decide whether a candidate model should be accepted.

## Train the final model

From the repository root, with the virtual environment activated and the dataset in place:

```powershell
python -m fraud_risk.train_final_model
```

This workflow trains the selected LightGBM model using the configured development data and saves the serialized model to:

```text
artifacts/fraud_model.joblib
```

Training a final model and evaluating a model are distinct operations. Verify the resulting artifact and its compatibility with the API before deploying a newly trained version.

## Evaluate model performance

Run the final evaluation workflow:

```powershell
python -m fraud_risk.run_final_evaluation
```

The workflow trains on development months 1–3 and evaluates on the reserved test partition, month 7.

### Recorded evaluation baseline

| Metric | Recorded result |
|---|---:|
| PR-AUC | 0.1976 |
| ROC-AUC | 0.8859 |
| Precision | 15.27% |
| Recall | 51.05% |
| True positives | 729 |
| False positives | 4,044 |
| False negatives | 699 |

These are historical project results. Reproduce them against the intended dataset, dependency versions, model artifact, and code revision before treating them as verified release metrics.

The configured decision threshold is **0.65** and remains provisional. At the recorded operating point, recall of 51.05% means that a substantial number of fraudulent transactions were missed, while precision of 15.27% means that many flagged transactions were legitimate.

Threshold selection should account for the relative cost of missed fraud, false alerts, and available review capacity. A threshold should not be selected solely to optimize one metric without considering the operational consequences.

## Candidate retraining and acceptance gate

The project includes a guarded candidate-retraining workflow:

```powershell
python -m fraud_risk.retraining
```

The workflow:

1. Loads the dataset and uses months 1–2 for candidate training.
2. Evaluates the candidate on validation month 3.
3. Reports validation PR-AUC, recall, and the configured minimum recall.
4. Saves the candidate to `artifacts/fraud_model_candidate.joblib` only if validation recall meets the configured minimum of **0.70**.
5. Rejects the candidate without saving it when the recall requirement is not met.

The gate uses the configuration in `selection_config.py`. Its purpose is to prevent a candidate that fails the recall requirement from being automatically treated as acceptable.

**Important limitations:**

- The gate does not prove that an accepted candidate is better than the currently deployed model across all relevant metrics.
- Passing the retraining-gate unit tests does not prove that a real training run will be accepted.
- A rejected candidate is a valid outcome and should be investigated rather than bypassed.
- Saving a candidate does not replace the active model or deploy anything.
- Promotion should be a separate, deliberate step involving model compatibility checks, evaluation, versioning, and deployment verification.
- Test month 7 must remain excluded from candidate selection and threshold tuning.

## Run the API locally

Ensure that the expected model artifact exists, then start the development server:

```powershell
python -m uvicorn fraud_risk.api:app --app-dir src --reload
```

The local service will usually be available at:

```text
http://127.0.0.1:8000
```

### Health check

Open:

http://127.0.0.1:8000/health

Expected response:

```json
{
  "status": "ok"
}
```

This confirms that the application responds. It does not, by itself, establish that the model artifact is available or that predictions are correct.

### Interactive API documentation

Visit:

http://127.0.0.1:8000/docs

Use the interactive documentation to submit a transaction to `POST /predict`. Supply the required feature fields with values matching the input schema expected by the model.

The endpoint returns:

- `fraud_probability`: the model's estimated probability of fraud.
- `fraud_prediction`: `1` if the estimated probability meets or exceeds the configured threshold; otherwise `0`.

These are model outputs, not definitive determinations of fraud.

### API error handling

The API validates request structure and feature inputs. Invalid requests should return an appropriate client error, while failures such as an unavailable model should be handled separately from successful predictions.

Use the automated tests to verify these behaviors after modifying the API or its input schema.

## Deployment

The application has been deployed to Render:

- **Service health:** https://fraud-risk-platform-2pb9.onrender.com/health
- **Interactive API documentation:** https://fraud-risk-platform-2pb9.onrender.com/docs
- **Prediction endpoint:** `POST https://fraud-risk-platform-2pb9.onrender.com/predict`
- **Metrics endpoint:** https://fraud-risk-platform-2pb9.onrender.com/metrics

These URLs identify the project's deployment, but service availability should be checked when needed. A free hosting tier may experience cold starts or other resource limitations.

The Dockerfile packages the API and its runtime dependencies. GitHub Actions runs the automated test workflow when changes are submitted. A successful CI run validates the configured tests; it does not replace deployment smoke tests or model-quality checks.

## Monitoring and drift reporting

### Prediction metrics

The `GET /metrics` endpoint exposes in-process monitoring information, including:

- Total prediction requests.
- Successful and failed prediction requests.
- Recorded request latency, including total and maximum latency.
- A summary of the latest saved drift report, when available.

These metrics are held in application memory and reset when the process restarts. They are not a persistent, centralized, or multi-worker monitoring solution.

### Run the drift check

From the repository root:

```powershell
python -m fraud_risk.run_drift_check
```

The workflow compares numeric feature distributions from training months 1–2 with validation month 3 and test month 7. It calculates the Population Stability Index (PSI) and writes the results to:

```text
reports/drift_report.json
```

The current investigation threshold is `0.2`. PSI is a heuristic for identifying distribution changes, not proof that model performance has degraded. A flagged feature should prompt investigation and should be interpreted alongside data quality, model performance, and domain knowledge.

The report represents **historical, offline drift analysis**. The API exposes its saved summary; it does not continuously calculate drift from live traffic. Distribution drift and predictive performance degradation are related but distinct concepts.

## Run tests

From the repository root, with the virtual environment activated:

```powershell
python -m compileall -q src
python -m pytest -v
```

The latest recorded local test run completed with **22 tests passing** and one non-blocking Starlette deprecation warning concerning the test client's HTTPX integration.

This is the recorded result, not a guarantee that every future code revision will pass. Rerun the commands after changes and review any new failures or warnings.

The retraining tests validate the acceptance-gate behavior using controlled test cases. They do not substitute for running the actual candidate-training workflow against the authorized dataset.

## Engineering limitations and next steps

The project demonstrates important ML engineering practices, but several safeguards would be needed before responsible production use:

1. **Model quality:** investigate the recorded recall and false-positive rate against documented business requirements.
2. **Threshold governance:** validate the threshold using the correct validation process and operational cost assumptions.
3. **Monitoring:** persist metrics, establish alerting, and distinguish infrastructure failures from data and model-quality issues.
4. **Live drift detection:** collect suitable reference and production data, calculate drift continuously or periodically, and investigate significant changes.
5. **Retraining orchestration:** schedule candidate training, record results and model versions, and require explicit promotion checks.
6. **Deployment safety:** verify artifacts, dependencies, configuration, health, and predictions before and after releases.
7. **Security and operations:** add appropriate authentication, rate limiting, secrets management, access controls, and operational incident procedures before exposing a service to sensitive real-world traffic.

## License and dataset provenance

No license or dataset redistribution permission should be assumed unless it is explicitly established in the repository. Review the relevant source terms and project licensing before redistributing the code, model, or data.

## Summary

This repository demonstrates an end-to-end fraud-risk ML engineering workflow: data preparation, model development, evaluation, API serving, containerization, automated testing, basic monitoring, offline drift analysis, and a candidate-retraining acceptance gate.

The goal is not simply to produce a trained model, but to demonstrate how its lifecycle components fit together, how their behavior can be tested, and where further validation and operational safeguards are required.
