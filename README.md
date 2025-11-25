# **Customer Churn Prediction – ANN Model (Streamlit Web App)**

This project is an interactive **Streamlit web application** that predicts customer churn using a trained Artificial Neural Network (ANN).
The model is built using **TensorFlow/Keras**, and the UI is built using **Streamlit**.

---

## 📦 **Project Structure**

```
📁 churn-app/
│
├── app.py
├── model.keras
├── scaler.pkl
├── gender_label_encoder.pkl
├── ohe_geography.pkl
├── requirements.txt
├── README.md
└── images/
      └── banner.png   (optional)
```

---

## 🧠 **Model Overview**

* **Model Type:** Artificial Neural Network (ANN)
* **Framework:** Keras / TensorFlow
* **Layers:**

  * Dense (64 neurons, ReLU)
  * Dense (32 neurons, ReLU)
  * Dense (1 neuron, Sigmoid)
* **Problem:** Binary Classification (Churn / No-Churn)

Model saved as: `model.keras`

---

## 🛠️ **Tech Stack**

| Layer    | Technology                   |
| -------- | ---------------------------- |
| Frontend | Streamlit                    |
| Backend  | Python                       |
| ML Model | TensorFlow / Keras           |
| Encoding | LabelEncoder + OneHotEncoder |
| Scaling  | StandardScaler               |



## 🧪 **Features in the App**

* Dropdowns for categorical features
* Sliders / number inputs for numerical features
* On-click prediction
* Clean UI
* Model probability output
* "Customer Will Churn / Not Churn" message
* Optional banner image

---

## 🧪 **Sample Input**

| Field         | Example |
| ------------- | ------- |
| Geography     | France  |
| Gender        | Male    |
| Age           | 45      |
| Credit Score  | 650     |
| Balance       | 120000  |
| Active Member | Yes     |

---

## 📤 **Sample Output**

```
Final Prediction: Customer is likely to churn ❌
Probability: 76.4%
```

or

```
Final Prediction: Customer will NOT churn ✅
Probability: 12.3%
```

---

## 📘 **Future Enhancements**

* Add Explainability (SHAP)
* Add charts & insights
* Add CSV bulk prediction
* Connect database for real-time data
* Deploy with Docker

---

## ⭐ **Support**

If this project helped you, please ⭐ star the repository!
