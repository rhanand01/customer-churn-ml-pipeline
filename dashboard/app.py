import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# API endpoint (FastAPI)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Path to dataset for analytics tab
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📉 Telecom Customer Churn Prediction")

tab_predict, tab_analytics = st.tabs(["🔮 Predict Churn", "📊 Analytics"])

# ------------------------------
# 🔮 TAB 1: Prediction Interface
# ------------------------------
with tab_predict:
    st.subheader("Enter Customer Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    with col2:
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["No internet service", "No", "Yes"])
        OnlineBackup = st.selectbox("Online Backup", ["No internet service", "No", "Yes"])

    with col3:
        DeviceProtection = st.selectbox("Device Protection", ["No internet service", "No", "Yes"])
        TechSupport = st.selectbox("Tech Support", ["No internet service", "No", "Yes"])
        StreamingTV = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"])
        StreamingMovies = st.selectbox("Streaming Movies", ["No internet service", "No", "Yes"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, max_value=300.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)

    if st.button("Predict Churn"):
        payload = {
            "gender": gender,
            "SeniorCitizen": SeniorCitizen,
            "Partner": Partner,
            "Dependents": Dependents,
            "tenure": tenure,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
        }

        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            result = response.json()
            proba = result["churn_probability"]
            label = result["churn_prediction"]

            st.markdown("---")
            st.subheader("Prediction Result")

            st.metric("Churn Probability", f"{proba * 100:.2f}%")

            if label == 1:
                st.error("⚠️ This customer is **likely to churn**.")
            else:
                st.success("✅ This customer is **likely to stay**.")

        except Exception as e:
            st.error(f"❌ Error calling API: {e}")


# ------------------------------
# 📊 TAB 2: Analytics
# ------------------------------
with tab_analytics:
    st.subheader("Dataset Insights")

    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}")
    else:
        df = pd.read_csv(DATA_PATH)

        col_a, col_b = st.columns(2)

        with col_a:
            st.write("Sample Data")
            st.dataframe(df.head())

        with col_b:
            st.write("Churn Distribution (%)")
            churn_percent = df["Churn"].value_counts(normalize=True) * 100
            st.bar_chart(churn_percent)

        st.markdown("### Churn vs Contract Type")
        contract_churn = (
            df.groupby(["Contract", "Churn"])
            .size()
            .reset_index(name="count")
            .pivot(index="Contract", columns="Churn", values="count")
            .fillna(0)
        )
        st.bar_chart(contract_churn)
