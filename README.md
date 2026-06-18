# 📊 Customer Churn Prediction ML App

A Machine Learning web application that predicts whether a customer will churn based on their service usage and account details.

Built using **Python, Scikit-Learn, and Streamlit**, and deployed on Streamlit Cloud.

---

## 🚀 Live Demo
👉 https://churn-prediction-ml-v.streamlit.app/

---

## 📌 Project Overview

Customer churn is a major problem for subscription-based businesses.  
This project builds a predictive ML model to identify customers likely to leave.

The system helps businesses:
- Reduce customer loss
- Improve retention strategies
- Make data-driven decisions

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression
- Framework: Scikit-learn Pipeline
- Preprocessing:
  - Handling missing values
  - One-hot encoding categorical features
  - Feature scaling (if applied in pipeline)

---

## 📊 Dataset

Telco Customer Churn Dataset:
- Customer demographics
- Account information
- Services subscribed
- Billing information

Target:
- `Churn` (Yes/No)

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Joblib

---

## 📂 Project Structure


Churn-Prediction-ML/
│
├── app.py # Streamlit web app
├── train.py # Model training script
├── models/
│ └── churn_model.pkl # Saved ML model
├── data/
│ └── churn.csv # Dataset
├── notebooks/ # EDA & experiments
├── requirements.txt # Dependencies
└── README.md


---

## 🚀 How to Run Locally

### 1. Clone repo
```bash
git clone https://github.com/OshadhaS/Churn-Prediction-ML.git
cd Churn-Prediction-ML
2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Run Streamlit app
streamlit run app.py

📈 Model Performance
Accuracy: ~79–82%
Balanced precision/recall for churn class
Tuned using preprocessing pipeline

🎯 Key Features
User-friendly web interface
Real-time churn prediction
Probability risk score
Handles categorical + numerical inputs

📸 UI Preview

(Add screenshots here later)

Example:

/screenshots/home.png
/screenshots/prediction.png
👨‍💻 Author
Oshadha
Machine Learning Project Portfolio
⭐ Future Improvements
Try XGBoost / Random Forest
Improve recall for churn class
Add database logging (customer predictions)
Deploy API version (FastAPI)

📌 Note

This is a beginner-to-intermediate level ML deployment project for learning and portfolio building.


---

# 💡 AFTER THIS (IMPORTANT)

Run:

```bash
git add README.md
git commit -m "add professional README"
git push
