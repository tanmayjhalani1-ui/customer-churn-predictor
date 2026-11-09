from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Churn_Modelling.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "churn_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "HasCrCard", "IsActiveMember", "EstimatedSalary"
]
CATEGORICAL_FEATURES = ["Geography", "Gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Exited"


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    required = set(FEATURES + [TARGET])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = frame[FEATURES + [TARGET]].copy()
    for column in NUMERIC_FEATURES + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Exited must contain only binary values 0 and 1.")
    if frame.empty or frame[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both target classes.")
    if (frame["Age"].dropna() <= 0).any():
        raise ValueError("Age values must be positive.")
    if (frame["Tenure"].dropna() < 0).any() or (frame["Balance"].dropna() < 0).any():
        raise ValueError("Tenure and Balance cannot be negative.")
    return frame


def build_pipeline(estimator: object) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocessing", preprocessing), ("classifier", estimator)])


def train(use_mlflow: bool = True) -> dict:
    data = load_dataset()
    X, y = data[FEATURES], data[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.07, max_iter=180, max_leaf_nodes=15,
            l2_regularization=1.0, random_state=42
        ),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        cv_scores[name] = {
            "mean_roc_auc": float(scores.mean()),
            "std_roc_auc": float(scores.std()),
        }
    best_name = max(cv_scores, key=lambda name: cv_scores[name]["mean_roc_auc"])
    model = build_pipeline(candidates[best_name])
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "selected_model": best_name,
        "dataset_rows": int(len(data)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "churn_rate": float(y.mean()),
        "features": FEATURES,
        "cross_validation": cv_scores,
        "holdout_metrics": metrics,
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "decision_threshold": 0.5,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if use_mlflow:
        mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
        mlflow.set_experiment("customer-churn-prediction")
        with mlflow.start_run(run_name=best_name):
            mlflow.log_params({
                "selected_model": best_name,
                "dataset_rows": len(data),
                "test_size": 0.2,
                "random_state": 42,
                "decision_threshold": 0.5,
            })
            mlflow.log_metrics({
                f"cv_{name}_mean_roc_auc": values["mean_roc_auc"]
                for name, values in cv_scores.items()
            })
            mlflow.log_metrics(metrics)
            mlflow.log_dict(metadata, "evaluation.json")
            mlflow.sklearn.log_model(model, artifact_path="model")
    print(json.dumps(metadata, indent=2))
    print("\nHoldout classification report")
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"\nSaved model to {MODEL_PATH}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and compare customer churn models.")
    parser.add_argument("--no-mlflow", action="store_true")
    arguments = parser.parse_args()
    train(use_mlflow=not arguments.no_mlflow)
