import streamlit as st
import numpy as np
from preprocess import preprocess_ekg
from model import predict

st.title("Klasifikasi Sinyal EKG")

uploaded_file = st.file_uploader("Upload file EKG (.txt)")

if uploaded_file:
    signal = np.loadtxt(uploaded_file)
    features = preprocess_ekg(signal)
    result = predict(features)

    st.success(f"Hasil klasifikasi: {result[0]}")
