import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pywt
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ==================== FUNGSI PREPROCESSING ====================

def dwt_denoise(signal, wavelet='db6', level=3):
    """Denoising dengan DWT"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = 0.3 * sigma * np.sqrt(2 * np.log(len(signal)))
    
    coeffs_thresh = [coeffs[0]]
    for i, c in enumerate(coeffs[1:], start=1):
        if i >= 3:
            coeffs_thresh.append(pywt.threshold(c, uthresh, mode='soft'))
        else:
            coeffs_thresh.append(c)
    
    return pywt.waverec(coeffs_thresh, wavelet)[:len(signal)]

def extract_features_from_window(window_signal):
    """Ekstraksi fitur dari sebuah window sinyal"""
    features = {
        'mean': np.mean(window_signal),
        'std': np.std(window_signal),
        'rms': np.sqrt(np.mean(window_signal**2)),
        'skewness': skew(window_signal),
        'kurtosis': kurtosis(window_signal),
        'max': np.max(window_signal),
        'min': np.min(window_signal),
        'ptp': np.ptp(window_signal),
        'energy': np.sum(window_signal**2)
    }
    return features

def preprocess_ecg_signal(signal, sampling_rate=250):
    """Preprocessing lengkap untuk sinyal ECG baru"""
    # 1. Denoising
    signal_filtered = dwt_denoise(signal)
    
    # 2. Windowing (2 detik dengan 50% overlap)
    window_sec = 2
    window_size = int(window_sec * sampling_rate)
    step_size = int(window_size / 2)
    
    windows = []
    for start in range(0, len(signal_filtered) - window_size + 1, step_size):
        segment = signal_filtered[start:start + window_size]
        windows.append(segment)
    
    # 3. Ekstraksi fitur dari setiap window
    features_list = []
    for window in windows:
        features = extract_features_from_window(window)
        features_list.append(features)
    
    # 4. Konversi ke DataFrame dan reshape untuk model
    df_features = pd.DataFrame(features_list)
    
    # Untuk Random Forest yang menggunakan sinyal mentah (flatten)
    # Jika model menggunakan fitur yang diekstrak, sesuaikan ini
    X_flat = np.array(windows).reshape(len(windows), -1)
    
    return df_features, X_flat, windows

# ==================== STREAMLIT APP ====================

def main():
    st.title("Classification of ECG Signals")
    st.write("Condition classification: Active vs Calm")
    
    # Sidebar untuk upload file
    st.sidebar.header("Upload ECG Data")
    uploaded_file = st.sidebar.file_uploader("Select the ECG file (.txt)", type=['txt'])
    
    # Pilihan: Input manual atau upload file
    option = st.sidebar.radio("Pilih input:", ["Upload File", "Manual Input"])
    
    if option == "Upload File" and uploaded_file is not None:
        # Baca file upload
        try:
            data = np.loadtxt(uploaded_file)
            st.success(f"File uploaded successfully! Sample count: {len(data)}")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    
    elif option == "Manual Input":
        # Input manual untuk demo
        st.sidebar.subheader("Parameter Sinyal Demo")
        signal_type = st.sidebar.selectbox("Tipe Sinyal", ["Normal", "Aritmia Ringan"])
        duration = st.sidebar.slider("Durasi (detik)", 5, 30, 10)
        
        # Generate sinyal demo
        t = np.linspace(0, duration, 250*duration)
        if signal_type == "Normal":
            data = np.sin(2*np.pi*1*t) + 0.5*np.sin(2*np.pi*2*t) + 0.1*np.random.randn(len(t))
        else:
            data = np.sin(2*np.pi*1.2*t) + 0.8*np.sin(2*np.pi*3*t) + 0.3*np.random.randn(len(t))
        
        st.info(f"Menggunakan sinyal demo ({signal_type}, {duration} detik)")
    
    else:
        st.warning("Please upload ECG file or select manual input")
        return
    
    
    # Tombol untuk proses
    if st.button("Classification Process"):
        with st.spinner("Processing ECG signals..."):
            try:
                # Load model yang sudah disimpan
                model = joblib.load("rf_model.pkl")
                
                # Preprocessing
                df_features, X_flat, windows = preprocess_ecg_signal(data)
                
                # Prediksi
                predictions = model.predict(X_flat)
                
                # Decode label
                le = LabelEncoder()
                le.fit(['bermain', 'tenang'])
                pred_labels = le.inverse_transform(predictions)
                
                # Hasil
                st.subheader("Classification Results")
                
                # Hitung persentase
                total_windows = len(pred_labels)
                count_bermain = np.sum(pred_labels == 'bermain')
                count_tenang = np.sum(pred_labels == 'tenang')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Active", f"{count_bermain} window", 
                             f"{(count_bermain/total_windows*100):.1f}%")
                with col2:
                    st.metric("Calm", f"{count_tenang} window", 
                             f"{(count_tenang/total_windows*100):.1f}%")
                
                
                
                # Kesimpulan
                st.subheader("Conclusion")
                if count_bermain > count_tenang:
                    st.success("✅ Dominant condition: ACTIVE (Active)")
                else:
                    st.success("✅ Dominant condition: CALM (Normal)")
                
            except Exception as e:
                st.error(f"Error dalam pemrosesan: {e}")
                st.info("Pastikan model 'rf_model.pkl' ada di direktori yang sama")

if __name__ == "__main__":
    main()
