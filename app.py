import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load the trained model and scaler
with open("bankruptcy_model.pkl", "rb") as model_file:
    model = joblib.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = joblib.load(scaler_file)

# Ensure correct feature names (fetch from the scaler)
expected_features = list(scaler.feature_names_in_)

# Streamlit UI
st.title("💰 Bankruptcy Prediction App 💼")
st.markdown("""
*🔎 Enter financial details below to predict if a company is at risk of bankruptcy.*  
📢 Disclaimer: This is an AI-based prediction and should not be considered financial advice.
""")

# Create input fields for only expected features
user_inputs = {}
for feature in expected_features:
    user_inputs[feature] = st.number_input(f"📊 {feature}", value=0.0, format="%.6f")

# Predict function
def predict_bankruptcy(input_features):
    input_data = pd.DataFrame([input_features], columns=expected_features)
    input_scaled = scaler.transform(input_data)  # ✅ No feature mismatch error now
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error("🔴 *HIGH RISK OF BANKRUPTCY!*\n📉 The company is in financial distress. Immediate action is required.")
    else:
        st.success("🟢 *FINANCIALLY STABLE*\n✅ The company appears financially sound with no immediate risk.")

# Button to trigger prediction
if st.button("🔍 Predict Bankruptcy"):
    predict_bankruptcy(list(user_inputs.values()))