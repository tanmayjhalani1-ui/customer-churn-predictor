import json

import pandas as pd
import streamlit as st

from src.inference import load_model, predict_customer
from src.train import DATA_PATH, METADATA_PATH

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")
st.title("Customer Churn Intelligence")
st.caption("Predict churn risk, explore customer patterns, and review model performance.")

try:
    model = load_model()
except FileNotFoundError:
    st.warning("Train the model before using predictions.")
    st.code("python -m src.train")
    st.stop()

metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

overview = st.columns(4)
overview[0].metric("Dataset records", f"{metadata.get('dataset_rows', 0):,}")
overview[1].metric("Selected model", metadata.get("selected_model", "Not available").replace("_", " ").title())
overview[2].metric("Holdout ROC-AUC", f"{metadata.get('holdout_metrics', {}).get('roc_auc', 0):.3f}")
overview[3].metric("Holdout PR-AUC", f"{metadata.get('holdout_metrics', {}).get('pr_auc', 0):.3f}")

prediction_tab, insights_tab, performance_tab = st.tabs(["Customer risk", "Portfolio insights", "Model performance"])

with prediction_tab:
    st.subheader("Assess an individual customer")
    with st.form("customer_form"):
        first, second, third = st.columns(3)
        with first:
            credit_score = st.number_input("Credit score", 300, 900, 650)
            age = st.number_input("Age", 18, 100, 35)
            tenure = st.number_input("Tenure (years)", 0, 10, 4)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        with second:
            balance = st.number_input("Account balance", 0.0, 250000.0, 60000.0, step=1000.0)
            products = st.selectbox("Number of products", [1, 2, 3, 4], index=0)
            gender = st.selectbox("Gender", ["Female", "Male"])
            salary = st.number_input("Estimated salary", 0.0, 200000.0, 75000.0, step=1000.0)
        with third:
            has_card = st.selectbox("Has credit card", ["Yes", "No"])
            active = st.selectbox("Active member", ["Yes", "No"])
        submitted = st.form_submit_button("Estimate churn risk", type="primary", use_container_width=True)
    if submitted:
        customer = {
            "CreditScore": credit_score, "Age": age, "Tenure": tenure,
            "Balance": balance, "NumOfProducts": products,
            "HasCrCard": int(has_card == "Yes"),
            "IsActiveMember": int(active == "Yes"),
            "EstimatedSalary": salary, "Geography": geography, "Gender": gender,
        }
        result = predict_customer(model, customer)
        a, b, c = st.columns(3)
        a.metric("Prediction", result["label"])
        b.metric("Churn probability", f"{result['churn_probability']:.1%}")
        c.metric("Risk band", result["risk_band"])
        st.progress(result["churn_probability"])
        st.caption("Risk estimates are model outputs, not certainty about an individual customer's future behavior.")

with insights_tab:
    st.subheader("Customer base overview")
    if DATA_PATH.exists():
        data = pd.read_csv(DATA_PATH)
        exited = data["Exited"].value_counts().rename(index={0: "Retained", 1: "Churned"})
        a, b, c = st.columns(3)
        a.metric("Customers", f"{len(data):,}")
        b.metric("Churn rate", f"{data['Exited'].mean():.1%}")
        c.metric("Average age", f"{data['Age'].mean():.1f}")
        st.write("Churn distribution")
        st.bar_chart(exited)
        st.write("Churn rate by geography")
        st.bar_chart(data.groupby("Geography")["Exited"].mean().sort_values(ascending=False))
        st.write("Churn rate by activity status")
        st.bar_chart(data.groupby("IsActiveMember")["Exited"].mean().rename(index={0: "Inactive", 1: "Active"}))
    else:
        st.info("Dataset not found.")

with performance_tab:
    st.subheader("Evaluation on held-out data")
    metrics = metadata.get("holdout_metrics", {})
    if metrics:
        metric_columns = st.columns(3)
        for column, key in zip(metric_columns, ["precision", "recall", "f1"]):
            column.metric(key.upper(), f"{metrics[key]:.3f}")
        st.write("Cross-validation ROC-AUC by model")
        cv = metadata.get("cross_validation", {})
        if cv:
            st.bar_chart(pd.Series({name: value["mean_roc_auc"] for name, value in cv.items()}))
        st.write("Confusion matrix")
        st.dataframe(pd.DataFrame(metadata["confusion_matrix"], index=["Actual stay", "Actual churn"], columns=["Predicted stay", "Predicted churn"]))
        st.caption("Model selection uses training-only cross-validation. Holdout metrics are reported separately.")
