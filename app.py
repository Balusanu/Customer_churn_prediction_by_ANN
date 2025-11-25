import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

scaler = pickle.load(open("scaler.pkl","rb"))
ohe = pickle.load(open("ohe.pkl","rb"))
gender_encoder = pickle.load(open("gender_label_encoder.pkl","rb"))
model = load_model("model.keras")
# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_input(data):

    # 1️⃣ Label Encode Gender
    data["Gender"] = gender_encoder.transform([data["Gender"]])[0]

    # 2️⃣ One Hot Encode Geography
    geo_ohe = ohe_geo.transform([[data["Geography"]]])
    geo_df = pd.DataFrame(geo_ohe, columns=ohe_geo.get_feature_names_out())

    # 3️⃣ Numerical data
    num_features = ["CreditScore", "Age", "Tenure", "Balance",
                    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"]

    num_data = pd.DataFrame([[data[f] for f in num_features]], columns=num_features)
    num_scaled = scaler.transform(num_data)
    num_df = pd.DataFrame(num_scaled, columns=num_features)

    # 4️⃣ Concatenate all features
    final_df = pd.concat([geo_df, num_df], axis=1)

    return final_df


# -------------------------------
# Prediction Function
# -------------------------------
def predict_churn(input_data):
    processed = preprocess_input(input_data)
    pred = model.predict(processed)[0][0]
    return float(pred)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏦 Customer Churn Prediction (ANN Model)")

CreditScore = st.number_input("Credit Score", 300, 900, 650)
Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
Gender = st.selectbox("Gender", ["Male", "Female"])
Age = st.number_input("Age", 18, 92, 35)
Tenure = st.number_input("Tenure (Years)", 0, 10, 3)
Balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
NumOfProducts = st.number_input("Number of Products", 1, 4, 1)
HasCrCard = st.selectbox("Has Credit Card", [1, 0])
IsActiveMember = st.selectbox("Is Active Member", [1, 0])
EstimatedSalary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict Churn"):
    test_input = {
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

    result = predict_churn(test_input)

    st.subheader("🔍 Result")
    if result > 0.5:
        st.error(f"❌ High Churn Probability: {result:.2f}")
    else:
        st.success(f"✔ Customer Likely to Stay: {result:.2f}")

