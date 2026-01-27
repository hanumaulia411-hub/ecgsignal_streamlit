import streamlit as st
import numpy as np
from preprocess import preprocess_ekg
from model import predict

st.title("Klasifikasi Sinyal EKG")

@st.cache_resource
def load_model():
    model = joblib.load('rf_model.pkl')  # Gunakan file model yang ada di repo
    return model

# Kemudian saat prediksi:
features = preprocess_ekg(data)  # Output dari preprocess.py yang sudah diperbaiki
prediction = model.predict(features)  # Prediksi dengan Random Forest
# prediction_proba = model.predict_proba(features)  # (Opsional) Untuk mendapatkan probabilitas

st.success(f"Hasil klasifikasi: {result[0]}")
