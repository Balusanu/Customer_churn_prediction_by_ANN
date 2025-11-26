import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ---------------------------------------------------------
# Load saved models
# ---------------------------------------------------------
model = load_model("model.keras")

with open("gender_label_encoder.pkl", "rb") as file:
    gender_encoder = pickle.load(file)

with open("ohe.pkl", "rb") as file:
    ohe = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Churn Prediction", page_icon="🌀", layout="centered")

st.title("🌀 Customer Churn Prediction App")
st.write("Provide customer details below to estimate churn probability using the trained ANN model.")

# ---------------------------------------------------------
# User Inputs
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    CreditScore = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18, max_value=95, value=40)
    Balance = st.number_input("Account Balance", min_value=0.0, value=60000.0)

with col2:
    Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    Tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)
    NumOfProducts = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
    EstimatedSalary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

HasCrCard = st.selectbox("Has Credit Card?", [0, 1])
IsActiveMember = st.selectbox("Is Active Member?", [0, 1])


# ---------------------------------------------------------
# Preprocessing function
# ---------------------------------------------------------
def preprocess_input():
    input_data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary
    }

    input_df = pd.DataFrame([input_data])

    # Encode Gender
    input_df["Gender"] = gender_encoder.transform(input_df["Gender"])

    # --------- SAFE One-Hot Encoding (FIXED) ---------
    geo_value = input_df["Geography"].iloc[0]
    geo_encoded = ohe.transform([[geo_value]])  # passing as list-of-list avoids NoneType.pop
    geo_df = pd.DataFrame(
        geo_encoded,
        columns=ohe.get_feature_names_out(["Geography"])
    )
    # --------------------------------------------------

    # Merge encoded geography
    input_df = pd.concat([input_df.drop(columns=["Geography"]), geo_df], axis=1)

    # Scale
    scaled = scaler.transform(input_df)

    return scaled


# ---------------------------------------------------------
# Predict Button
# ---------------------------------------------------------
if st.button("Predict Churn"):
    try:
        processed = preprocess_input()

        prediction = model.predict(processed)[0][0]
        churn = prediction > 0.5

        st.subheader("🔍 Prediction Result")
        st.write(f"**Churn Probability:** `{prediction:.4f}`")

        if churn:
            st.error("🚨 The customer is likely to CHURN.")
        else:
            st.success("🟩 The customer is NOT likely to churn.")

    except Exception as e:
        st.error(f"Error: {e}")
