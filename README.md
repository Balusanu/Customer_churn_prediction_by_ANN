# 🌟 **Customer Churn Prediction using Artificial Neural Networks**

A complete end-to-end **Deep Learning project** that predicts whether a bank customer is likely to churn, built using **TensorFlow**, **Streamlit**, and **Scikit-Learn**.

This repository includes:

* Data preprocessing
* ANN model training
* Saving encoders and scalers
* Full inference pipeline
* A clean, production-ready Streamlit web app

---

## 📌 **1. Project Overview**

Customer churn is a critical challenge for businesses. Predicting *which* customers may leave allows teams to take proactive retention action.

This project uses an **Artificial Neural Network (ANN)** to classify whether a customer will churn based on features such as credit score, age, geography, account balance, and more.

The Streamlit app lets you input customer attributes and instantly receive a churn prediction powered by the trained ANN.

---

## 📊 **2. Dataset**

The project uses the **Churn_Modelling.csv** dataset, containing 10,000 customer records with:

* Credit score
* Geography
* Gender
* Age
* Tenure
* Balance
* Number of products
* Credit card status
* Active member status
* Estimated salary
* Churn (target)

---

## 🧹 **3. Data Preprocessing Pipeline**

Applied transformations include:

### ✔ Dropping unnecessary columns

* `RowNumber`, `CustomerId`, `Surname`

### ✔ Encoding

* **LabelEncoder** → Gender
* **OneHotEncoder (with handle_unknown='ignore')** → Geography
* Saved as:

  * `gender_label_encoder.pkl`
  * `ohe.pkl`

### ✔ Scaling

* **StandardScaler** used on all features
* Saved as:

  * `scaler.pkl`

### ✔ Train-test split

`train_test_split(test_size=0.2, random_state=42)`

---

## 🧠 **4. ANN Model Architecture**

The neural network is built using **TensorFlow Keras**:

| Layer | Units | Activation |
| ----- | ----- | ---------- |
| Dense | 64    | ReLU       |
| Dense | 32    | ReLU       |
| Dense | 1     | Sigmoid    |

### Loss / Optimizer

* Loss: **Binary Crossentropy**
* Optimizer: **Adam**
* Metrics: **Accuracy**

### Training Tools

* Early Stopping
* TensorBoard Logs
* Model saved as `model.keras`

---

## 🚀 **5. Streamlit App**

A complete web interface that performs:

* Gender encoding
* Geography one-hot encoding (bug-free, safe implementation)
* Scaling using saved scaler
* ANN prediction
* Displays churn probability and final label

---

## 📁 **7. Repository Structure**

```
📦 Customer-Churn-ANN
│
├── app.py                    # Streamlit web app
├── model.keras               # Trained ANN model
├── scaler.pkl                # StandardScaler
├── ohe.pkl                   # OneHotEncoder
├── gender_label_encoder.pkl  # LabelEncoder for gender
├── Churn_Modelling.csv       # Dataset
├── requirements.txt
└── README.md                 # Documentation
```

---

## 🎯 **8. Example Prediction Flow**

1. User inputs customer details
2. App encodes & scales data
3. ANN predicts churn probability
4. App displays:

   * Probability score
   * Final label (“Likely to churn” / “Not likely to churn”)

---

## 📦 **9. Requirements**

Key Python libraries:

* tensorflow
* streamlit
* numpy
* pandas
* scikit-learn
* pickle

You can install all dependencies using:

```
pip install -r requirements.txt
```

---

