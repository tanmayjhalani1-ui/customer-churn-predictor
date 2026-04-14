import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Customer Churn Prediction")

st.title("Customer Churn Prediction")

df = pd.read_csv("Churn_Modelling.csv")

df = df.drop(columns=['RowNumber','CustomerId','Surname'])

df = pd.get_dummies(df, columns=['Geography','Gender'], drop_first=True)

X = df.drop("Exited", axis=1)
y = df["Exited"]

@st.cache_resource
def train_model():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

if "ready" not in st.session_state:
    st.session_state.ready = False

if st.button("Train Model"):
    model = train_model()
    st.session_state.model = model
    st.session_state.ready = True
    st.success("Model trained")

if st.session_state.ready:

    st.subheader("Enter Customer Details")

    credit_score = st.number_input("Credit Score", 300, 900, 600)
    age = st.number_input("Age", 18, 100, 30)
    tenure = st.number_input("Tenure", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
    num_products = st.number_input("Number of Products", 1, 4, 1)
    has_cr_card = st.selectbox("Has Credit Card", [0, 1])
    is_active = st.selectbox("Is Active Member", [0, 1])
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])

    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0
    gender_male = 1 if gender == "Male" else 0

    if st.button("Predict"):

        input_data = np.array([[credit_score, age, tenure, balance,
                                num_products, has_cr_card, is_active,
                                estimated_salary, geo_germany,
                                geo_spain, gender_male]])

        prediction = st.session_state.model.predict(input_data)[0]

        if prediction == 1:
            st.error("Customer will leave")
        else:
            st.success("Customer will stay")

else:
    st.info("Train the model first")