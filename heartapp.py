import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("heart_random.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Heart Disease Prediction App")
st.write("Provide the following health metrics to predict the risk of heart disease.")

# Input form
age = st.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholestoral in mg/dl (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", value=1.0, format="%.1f")
slope = st.selectbox("Slope of the Peak Exercise ST Segment (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (ca)", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Prepare input
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                        thalach, exang, oldpeak, slope, ca, thal]])
columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
input_df = pd.DataFrame(input_data, columns=columns)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🚨 The model predicts a **high risk of heart disease.** Please consult a doctor.")
    else:
        st.info("✅ The model predicts a **low risk of heart disease.**")

st.markdown("---")
st.caption("Model: Random Forest Classifier | Dataset: heart.csv")
