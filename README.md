# Fraud Risk Platform

A Python machine-learning platform for detecting potentially fraudulent transactions using LightGBM, with a reusable prediction pipeline, a FastAPI service, automated tests, and initial monitoring and drift-detection capabilities.

## Project status

The core training, evaluation, saved-model prediction, API testing, Docker packaging workflow, and automated test workflow have been exercised. The project includes initial prediction metrics and historical feature-drift reporting, but remains **not production-ready** pending further validation, operational safeguards, and deployment testing.

## Features

- LightGBM-based fraud classification.
- Data loading, validation, preprocessing, and chronological data splitting.
- Model training, artifact serialization, and final test evaluation.
- Reusable prediction pipeline with a configurable decision threshold.
- FastAPI prediction service with input validation and error handling.
- Docker-based application packaging.
- GitHub Actions continuous integration test workflow.
- Prediction request counts, success/failure counts, and latency metrics.
- Numeric feature drift detection using Population Stability Index (PSI).
- JSON drift reports exposed through the `/metrics` endpoint.
- Automated tests covering API behavior, data splitting, preprocessing, drift detection, and drift-report generation.

## Technology stack

- Python 3.13+
- pandas
- scikit-learn
- LightGBM
- joblib
- FastAPI and Uvicorn
- pytest
- Docker
- GitHub Actions

## Repository structure

```text
fraud-risk-platform/
├── data/
│   └── raw/
│       ├── CCFraud_data.csv
│       └── DATA_PROVENANCE.md
├── reports/
│   ├── final_test_results.md
│   └── drift_report.json
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
│   └── test_preprocessing.py
├── artifacts/              # Generated model; not tracked by Git
├── .github/                # GitHub Actions workflows
├── Dockerfile
├── pyproject.toml
└── README.md
```

The tree shows the main project files; supporting configuration and other repository files may also be present.

## Installation

Requires Python 3.13 or later.

Clone the repository and enter the project directory:

```bash
git clone https://github.com/increase-codes/fraud-risk-platform.git
cd fraud-risk-platform
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest
```

If Python 3.13 is already available through the `python` command, you can use `python -m venv .venv` instead.

## Dataset setup

The training and evaluation workflows expect the CSV at:

```text
data/raw/CCFraud_data.csv
```

The dataset is not included in this repository because redistribution permissions have not been verified. Obtain it from an authorized source and review `data/raw/DATA_PROVENANCE.md` before use.

The CSV must contain the columns expected by the project's data-processing and modeling modules, including the fraud target `fraud_bool` and temporal column `month`.

The project uses a chronological split:

- **Training:** months 1 and 2
- **Validation:** month 3
- **Test:** month 7

The reserved test partition should remain separate from model selection and threshold tuning.

## Train the final model

From the repository root, with the virtual environment activated:

```powershell
python -m fraud_risk.train_final_model
```

The script trains the selected LightGBM pipeline using the configured development data and saves the model to:

```text
artifacts/fraud_model.joblib
```

This artifact is generated locally and is excluded from Git. Generate it again when setting up a fresh environment.

## Evaluate model performance

Run:

```powershell
python -m fraud_risk.run_final_evaluation
```

This workflow trains using development months 1–3 and evaluates against the reserved test month, month 7.

### Recorded evaluation baseline

| Metric | Result |
|---|---:|
| PR-AUC | 0.1976 |
| ROC-AUC | 0.8859 |
| Precision | 15.27% |
| Recall | 51.05% |
| True positives | 729 |
| False positives | 4,044 |
| False negatives | 699 |

These are historical project results. Reproduce them against the intended dataset and code version before treating them as verified release metrics.

The current decision threshold is **0.65** and remains provisional. The precision and false-positive count indicate that many flagged transactions are legitimate. Threshold selection should reflect operational costs, the relative cost of missed fraud, and available fraud-review capacity.

## Run the API

First, ensure the trained model artifact exists. Start the service from the repository root:

```powershell
python -m uvicorn fraud_risk.api:app --app-dir src --reload
```

The development server will usually be available at `http://127.0.0.1:8000`.

### Health check

Open:

```text
http://127.0.0.1:8000/health
```

Expected response:

```json
{"status": "ok"}
```

This endpoint confirms that the application responds; it does not independently verify model availability or prediction correctness.

### Interactive API documentation

Visit:

```text
http://127.0.0.1:8000/docs
```

Use the interactive documentation to submit a transaction to `POST /predict`. Supply the complete set of model input features, with numeric and categorical values matching the expected input schema.

The endpoint returns:

- `fraud_probability`: the model's estimated fraud probability.
- `fraud_prediction`: `1` if the probability meets or exceeds the configured threshold, otherwise `0`.

The prediction is a model output, not a definitive determination of fraud.

## Monitoring and drift reporting

### Prediction metrics

The `GET /metrics` endpoint reports in-process prediction monitoring metrics, including:

- Total prediction requests.
- Successful and failed prediction requests.
- Total, average, and maximum request latency.
- A summary of the latest saved drift report, when available.

These metrics are held in application memory and reset when the process restarts. They are not yet a persistent, centralized, or multi-worker monitoring solution.

### Run the drift check

From the repository root, run:

```powershell
python -m fraud_risk.run_drift_check
```

The workflow compares numeric feature distributions from training months 1–2 against validation month 3 and test month 7. It calculates the Population Stability Index (PSI) and writes the results to:

```text
reports/drift_report.json
```

The current investigation threshold is `0.2`. PSI is a heuristic for identifying distribution changes, not proof that model performance has degraded. Thresholds should be interpreted in context and validated for the intended application.

The report currently represents **historical, offline drift analysis**. The API exposes its saved summary; it does not continuously calculate drift from live traffic.

## Run tests

From the repository root:

```powershell
python -m compileall -q src
python -m pytest -v
```

The latest recorded local test run