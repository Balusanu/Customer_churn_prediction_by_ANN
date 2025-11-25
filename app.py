import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# -----------------------------
# Load Model & Preprocessors
# -----------------------------
model = load_model("model.keras",compile=False)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("ohe.pkl", "rb") as f:
    ohe = pickle.load(f)

with open("gender_label_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

# -----------------------------
# FINAL TRAINING COLUMN ORDER
# -----------------------------
final_columns = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

# ---------------------------------------------------------
# Function: Preprocess Input EXACTLY like training phase
# ---------------------------------------------------------
def preprocess_input(input_data):

    # Convert dict → DataFrame
    df = pd.DataFrame([input_data])

    # Encode Gender
    df["Gender"] = gender_encoder.transform(df["Gender"])

    # OHE for Geography
    geo_ohe = ohe.transform(df[["Geography"]])
    geo_df = pd.DataFrame(geo_ohe, columns=ohe.get_feature_names_out(["Geography"]))

    # Drop original column
    df = df.drop(columns=["Geography"])

    # Combine numerical + ohe columns
    df = pd.concat([df, geo_df], axis=1)

    # ---- FIX FEATURE MISMATCH ----
    # Add missing columns & reorder
    for col in final_columns:
        if col not in df.columns:
            df[col] = 0  # add missing columns (usually one-hot)

    df = df[final_columns]  # reorder correctly

    # Scale numeric data
    df_scaled = scaler.transform(df)

    return df_scaled


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🔮 Customer Churn Prediction App (ANN Model)")
st.write("Enter customer details to predict churn probability.")

# -----------------------------
# Input Widgets
# -----------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_crcard = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("Predict Churn"):

    input_data = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_crcard,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }

    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0][0]

    st.subheader("📌 Prediction Result")
    st.write(f"**Churn Probability:** `{prediction:.4f}`")

    if prediction > 0.5:
        st.error("🚨 The customer is **likely to churn**.")
    else:
        st.success("✅ The customer is **not likely to churn**.")
