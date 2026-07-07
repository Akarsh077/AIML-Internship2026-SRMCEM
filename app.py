import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
# Load trained model
model = tf.keras.models.load_model("wine_model.h5")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Wine Quality Predictor\nAkarsh07", page_icon="🍷")

st.title("🍷 Wine Quality Prediction")
st.write("Enter the wine characteristics below and click Predict.")

fixed_acidity = st.number_input("Fixed Acidity", value=7.4)
volatile_acidity = st.number_input("Volatile Acidity", value=0.70)
citric_acid = st.number_input("Citric Acid", value=0.00)
residual_sugar = st.number_input("Residual Sugar", value=1.9)
chlorides = st.number_input("Chlorides", value=0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0)
density = st.number_input("Density", value=0.9978)
pH = st.number_input("pH", value=3.51)
sulphates = st.number_input("Sulphates", value=0.56)
alcohol = st.number_input("Alcohol", value=9.4)

if st.button("Predict Quality"):

    sample = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)
    predicted_quality = np.argmax(prediction, axis=1)[0] + 3

    st.success(f"Predicted Wine Quality: {predicted_quality}")