<div align="center">
  <img src="assets/portfolio-banner.svg" alt="Customer Churn Prediction — Tanmay Jhalani" width="100%" />
</div>

<div align="center">
  <strong>Designed and engineered by Tanmay Jhalani</strong> · <a href="https://github.com/tanmayjhalani1-ui">GitHub Profile</a>
</div>

---

# Customer Churn Prediction

An end-to-end customer churn machine learning application built around the original Churn Modelling dataset and prediction project. The repository now includes a reproducible training pipeline, model comparison, MLflow experiment tracking, an interactive Streamlit dashboard, a lightweight FastAPI service, and Docker support.

## Capabilities

- Data validation and feature selection.
- Preprocessing with scikit-learn pipelines.
- Comparison of Logistic Regression, Random Forest, and Histogram Gradient Boosting.
- Stratified five-fold cross-validation for model selection.
- Holdout evaluation using ROC-AUC, PR-AUC, precision, recall, F1, accuracy, and a confusion matrix.
- MLflow tracking for model parameters, metrics, metadata, and fitted model.
- Streamlit dashboard for individual churn-risk predictions and portfolio insights.
- FastAPI health and prediction endpoints.
- Docker and Docker Compose.
- GitHub Actions workflow for dependency installation, compilation, and model-training validation.

## Repository structure

```text
.
├── api/
│   └── main.py
├── src/
│   ├── inference.py
│   └── train.py
├── .github/workflows/ci.yml
├── app.py
├── customerchurn.py
├── customerchurn.ipynb
├── Churn_Modelling.csv
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

The original Streamlit script, notebook, and dataset are retained. The new structured application is available through `app.py`.

## Quick start

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and train the models:

```bash
pip install -r requirements.txt
python -m src.train
```

The training command validates the source data, compares candidate estimators using cross-validation on the training split, selects the best mean ROC-AUC candidate, evaluates it on a held-out test set, and saves the fitted pipeline and metadata under `artifacts/`.

## Launch the dashboard

```bash
streamlit run app.py
```

The dashboard includes customer-level churn estimates, portfolio churn summaries, and model evaluation details.

## Launch the API

```bash
uvicorn api.main:app --reload
```

- Health endpoint: `http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

### Example prediction request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "credit_score": 650,
    "age": 35,
    "tenure": 4,
    "balance": 60000,
    "num_products": 2,
    "has_credit_card": true,
    "is_active_member": true,
    "estimated_salary": 75000,
    "geography": "France",
    "gender": "Female"
  }'
```

The response includes a predicted class, churn probability, and a simple risk band. These are model estimates and should not be treated as certainty about an individual customer.

## MLflow experiments

Training runs are recorded in the local `mlruns/` directory. Start the tracking UI with:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Open `http://127.0.0.1:5000` to compare model runs, evaluation metrics, and logged artifacts.

## Docker

Train the model once, then run both services:

```bash
docker compose up --build
```

- Streamlit dashboard: `http://localhost:8501`
- FastAPI docs: `http://localhost:8000/docs`

The generated model artifact is mounted into both containers.

## Modeling approach

The target is `Exited`. Identifier columns such as `RowNumber`, `CustomerId`, and `Surname` are excluded from model features. Numeric features are median-imputed and standardized; categorical features are mode-imputed and one-hot encoded. All preprocessing is part of the fitted scikit-learn pipeline, so transformations are learned from training data rather than the full dataset.

A stratified holdout split is reserved for final evaluation. Candidate models are compared using stratified five-fold ROC-AUC on the training portion only. Churn is typically the minority class, so the project reports PR-AUC, precision, and recall in addition to accuracy and ROC-AUC.

## Responsible use and limitations

- Predictions represent statistical associations in the supplied dataset, not causal explanations.
- The dataset's coverage and representativeness may not match a real bank's current customer population.
- Risk bands are simple thresholds for demonstration, not calibrated business decision policies.
- Before operational use, validate on recent representative data, assess subgroup performance and calibration, establish a decision policy with business stakeholders, and monitor drift and outcomes.

## Original work

The original `customerchurn.py`, `customerchurn.ipynb`, and `Churn_Modelling.csv` are preserved. The end-to-end application is added alongside the original files.
