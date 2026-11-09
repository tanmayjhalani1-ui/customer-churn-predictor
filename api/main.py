from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import load_model, predict_customer

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Lightweight API for customer churn risk inference.",
    version="1.0.0",
)
model = None


class CustomerFeatures(BaseModel):
    credit_score: int = Field(ge=300, le=900)
    age: int = Field(gt=0, le=120)
    tenure: int = Field(ge=0, le=10)
    balance: float = Field(ge=0)
    num_products: int = Field(ge=1, le=4)
    has_credit_card: bool
    is_active_member: bool
    estimated_salary: float = Field(ge=0)
    geography: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]


@app.on_event("startup")
def load_artifact():
    global model
    try:
        model = load_model()
    except FileNotFoundError:
        model = None


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(payload: CustomerFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not trained. Run python -m src.train first.")
    customer = {
        "CreditScore": payload.credit_score,
        "Age": payload.age,
        "Tenure": payload.tenure,
        "Balance": payload.balance,
        "NumOfProducts": payload.num_products,
        "HasCrCard": int(payload.has_credit_card),
        "IsActiveMember": int(payload.is_active_member),
        "EstimatedSalary": payload.estimated_salary,
        "Geography": payload.geography,
        "Gender": payload.gender,
    }
    return predict_customer(model, customer)
