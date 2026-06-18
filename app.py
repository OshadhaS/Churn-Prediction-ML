import streamlit as st
import joblib
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="centered")

# ---------------- LOAD MODEL ----------------
model = joblib.load("models/churn_model.pkl")

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ About App")
st.sidebar.info("""
This AI app predicts whether a customer will churn or not.

Built using:
- Python
- Scikit-learn
- Streamlit
""")

# ---------------- HEADER ----------------
st.markdown("""
# 📊 Customer Churn Prediction App
Predict whether a customer will leave or stay using Machine Learning.
""")

st.write("Fill customer details below:")

# ---------------- INPUTS ----------------
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.number_input("Tenure", min_value=0)

phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

payment = st.selectbox("Payment Method", [
    "Electronic check",
    "Mailed check",
    "Bank transfer (automatic)",
    "Credit card (automatic)"
])

monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

# ---------------- FEATURE ENGINEERING ----------------
total_charges = monthly_charges * tenure

# ---------------- PREDICTION ----------------
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(input_data)

    # Probability
    try:
        prob = model.predict_proba(input_data)[0][1]
        st.progress(int(prob * 100))
        st.write(f"⚡ **Churn Probability: {prob:.2%}**")
    except:
        st.warning("Probability not available for this model")

    # Result
    if prediction[0] == 1:
        st.error("⚠ HIGH RISK: Customer WILL churn")
    else:
        st.success("✅ LOW RISK: Customer will NOT churn")