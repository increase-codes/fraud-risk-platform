# Fraud Risk Platform

A Python machine-learning platform for detecting potentially fraudulent transactions using LightGBM, with a reusable prediction pipeline and a FastAPI service.

## Project status

The model training workflow, prediction API, and core tests are implemented. The project is still undergoing final validation and is **not yet production-ready**.

## Features

* LightGBM-based fraud classification.
* Data loading, validation, preprocessing, and temporal data splitting.
* Model training and artifact serialization.
* Final evaluation on a reserved test month.
* REST API for transaction fraud predictions.
* Automated tests for data splitting and preprocessing.

## Technology stack

* Python 3.13+
* pandas
* scikit-learn
* LightGBM
* joblib
* FastAPI and Uvicorn
* pytest for testing

## Repository structure

```text
fraud-risk-platform/
├── data/
│   └── raw/
│       ├── CCFraud_data.csv
│       └── DATA_PROVENANCE.md
├── src/
│   └── fraud_risk/
│       ├── api.py
│       ├── config.py
│       ├── data_loader.py
│       ├── data_split.py
│       ├── data_validation.py
│       ├── eda.py
│       ├── evaluation.py
│       ├── model.py
│       ├── predict.py
│       ├── preprocessing.py
│       ├── run_evaluation.py
│       ├── run_final_evaluation.py
│       ├── selection_config.py
│       └── train_final_model.py
├── tests/
├── artifacts/              # Generated model; not tracked by Git
├── pyproject.toml
└── README.md
```

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

## Train the final model

From the repository root, with the virtual environment activated:

```powershell
python -m fraud_risk.train_final_model
```

The script trains the selected LightGBM pipeline using the configured training and validation partitions, then saves the model to:

```text
artifacts/fraud_model.joblib
```

This artifact is generated locally and is excluded from Git. Generate it again when setting up a fresh environment.

## Evaluate model performance

Run:

```powershell
python -m fraud_risk.run_final_evaluation
```

This workflow fits a model using development months 1–3 and evaluates it on the reserved test month, month 7. Keep the test partition separate from model and threshold selection.

Recorded evaluation baseline:

| Metric          | Result |
| --------------- | -----: |
| PR-AUC          | 0.1976 |
| ROC-AUC         | 0.8859 |
| Precision       | 15.27% |
| Recall          | 51.05% |
| True positives  |    729 |
| False positives |  4,044 |
| False negatives |    699 |

These are historical project results and should be reproduced against the intended dataset and code version before being treated as verified release metrics.

The current decision threshold is 0.65 and remains provisional. The precision and false-positive count indicate that many flagged transactions are legitimate. Threshold selection should reflect operational costs and fraud-review capacity.

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

### Interactive API documentation

Visit:

```text
http://127.0.0.1:8000/docs
```

Use the interactive documentation to submit a transaction to `POST /predict`. Supply the complete set of model input features, with numeric and categorical values matching the training schema.

The endpoint returns:

* `fraud_probability`: the model's estimated fraud probability.
* `fraud_prediction`: `1` if the probability meets or exceeds the configured threshold, otherwise `0`.

The prediction is a model output, not a definitive determination of fraud.

## Run tests

From the repository root:

```powershell
python -m compileall -q src
python -m pytest -v
```

The current core test suite contains six tests covering temporal data splitting and preprocessing. Passing these tests does not replace API integration, input-validation, or deployment testing.

## Important limitations

* The decision threshold has not been established as optimal for real operational costs.
* Performance has been evaluated on the reserved test month; broader temporal and out-of-distribution validation remains necessary.
* API input validation, error handling, and operational safeguards require further review.
* The API is currently a development service; production deployment requires appropriate security, monitoring, logging, and resource management.
* Model artifacts are generated locally rather than distributed through Git.

## License and data use

Review the applicable dataset license and redistribution conditions before using or sharing the data. Add a project license only after deciding the intended licensing terms for the code.

## Development status

The project is in its integration and finalization stage. The next milestones are documentation verification, stronger API tests, reproducibility checks, continuous integration, and deployment readiness.
