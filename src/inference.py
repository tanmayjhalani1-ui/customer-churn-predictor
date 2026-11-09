from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"
FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary", "Geography", "Gender"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model is not trained. Run: python -m src.train")
    return joblib.load(MODEL_PATH)


def predict_customer(model, customer: dict) -> dict:
    row = pd.DataFrame([{key: customer[key] for key in FEATURES}], columns=FEATURES)
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    return {
        "prediction": prediction,
        "label": "Likely to churn" if prediction else "Likely to stay",
        "churn_probability": probability,
        "risk_band": "High" if probability >= 0.65 else "Medium" if probability >= 0.35 else "Low",
    }
