import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")

# ------------------------
# USER INPUTS (SIMPLIFIED)
# ------------------------

age = st.slider("Age", 18, 100, 45)
trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 120)
chol = st.slider("Cholesterol (mg/dl)", 100, 400, 220)
thalch = st.slider("Max Heart Rate", 60, 200, 150)
oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
ca = st.slider("Number of Blocked Vessels (0–3)", 0, 3, 0)

sex = st.radio("Sex", ["Male", "Female"])

cp = st.selectbox("Chest Pain Type", [
    "No Chest Pain",
    "Mild Chest Discomfort",
    "Chest Pain During Activity",
    "Heavy Chest Pain"
])

cp_map = {
    "No Chest Pain": "asymptomatic",
    "Mild Chest Discomfort": "atypical angina",
    "Chest Pain During Activity": "non-anginal",
    "Heavy Chest Pain": "typical angina"
}

cp_value = cp_map[cp]

fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
restecg = st.selectbox("Rest ECG", ["Normal ECG", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
slope = st.selectbox("Heart Rate Slope", ["Downsloping", "Flat", "Upsloping"])
thal = st.selectbox("Thalassemia Result", ["Fixed Defect", "Normal", "Reversible Defect"])

# ------------------------
# FEATURE ENGINEERING
# ------------------------
chol_per_age = chol / age
bp_per_age = trestbps / age
hr_reserve = 220 - age - thalch
chol_bp_ratio = chol / (trestbps + 1)
chol_high = 1 if chol > 240 else 0

# ------------------------
# BUILD INPUT FEATURE ROW
# ------------------------
row = {
    'age': age,
    'trestbps': trestbps,
    'chol': chol,
    'thalch': thalch,
    'oldpeak': oldpeak,
    'ca': ca,
    'chol_per_age': chol_per_age,
    'bp_per_age': bp_per_age,
    'hr_reserve': hr_reserve,
    'chol_bp_ratio': chol_bp_ratio,
    'chol_high': chol_high,

    'sex_Male': 1 if sex == "Male" else 0,

    'cp_atypical angina': 1 if cp_value == "atypical angina" else 0,
    'cp_non-anginal': 1 if cp_value == "non-anginal" else 0,
    'cp_typical angina': 1 if cp_value == "typical angina" else 0,
    # asymptomatic case = all 0

    'fbs_True': 1 if fbs == "Yes" else 0,

    'restecg_normal': 1 if restecg == "Normal ECG" else 0,
    'restecg_st-t abnormality': 1 if restecg == "ST-T Abnormality" else 0,

    'exang_True': 1 if exang == "Yes" else 0,

    'slope_flat': 1 if slope == "Flat" else 0,
    'slope_upsloping': 1 if slope == "Upsloping" else 0,

    'thal_normal': 1 if thal == "Normal" else 0,
    'thal_reversable defect': 1 if thal == "Reversible Defect" else 0
}

cols = list(row.keys())

df = pd.DataFrame([row], columns=cols)

# ------------------------
# SCALE & PREDICT
# ------------------------
scaled = scaler.transform(df)

if st.button("Predict"):
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] * 100

    if pred == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob:.1f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob:.1f}%)")
