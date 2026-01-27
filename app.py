# app.py
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from preprocess import preprocess_ekg

# Title
st.title("🫀 Klasifikasi Sinyal ECG: Aktif vs Tenang")
st.markdown("Upload sinyal ECG dalam format CSV untuk diklasifikasikan")

# Initialize session state untuk menyimpan data
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Sidebar untuk upload file
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV",
    type=['csv', 'txt'],
    help="File harus berisi sinyal ECG dalam satu kolom"
)

# Load model dengan caching
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fungsi untuk membaca file
def read_uploaded_file(file):
    try:
        # Coba baca sebagai CSV
        df = pd.read_csv(file, header=None)
        # Ambil kolom pertama sebagai sinyal
        data = df.iloc[:, 0].values
        return data
    except Exception as e:
        st.error(f"Error membaca file: {e}")
        return None

# Main flow aplikasi
def main():
    model = load_model()
    
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file rf_model.pkl ada di repository.")
        return
    
    # Jika ada file yang diupload
    if uploaded_file is not None:
        # Baca file
        data = read_uploaded_file(uploaded_file)
        
        if data is not None:
            # Simpan di session state
            st.session_state.uploaded_data = data
            
            # Tampilkan preview data
            st.subheader("📊 Preview Sinyal ECG")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Jumlah sampel:** {len(data)}")
                st.write(f"**Nilai min/max:** {np.min(data):.4f} / {np.max(data):.4f}")
                st.write(f"**Rata-rata:** {np.mean(data):.4f}")

            with col2:
                # Tentukan berapa banyak data yang akan ditampilkan
                display_count = min(10, len(data))
                # Buat dataframe dengan jumlah yang sama
                df_preview = pd.DataFrame({
                    'Sample': range(1, display_count + 1),
                    'Value': data[:display_count]
                })
                st.dataframe(df_preview, height=200)
            
            # Visualisasi sinyal (500 sample pertama)
            fig = go.Figure()
            display_samples = min(500, len(data))
            fig.add_trace(go.Scatter(
                x=list(range(display_samples)),
                y=data[:display_samples],
                mode='lines',
                line=dict(color='blue', width=2),
                name='Sinyal ECG'
            ))
            fig.update_layout(
                title="Visualisasi Sinyal ECG",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tombol untuk klasifikasi
            if st.button("🚀 Mulai Klasifikasi", type="primary", use_container_width=True):
                with st.spinner("Memproses sinyal..."):
                    try:
                        # TAMPILKAN INFO WINDOW
                        st.info(f"📏 Menggunakan window: 500 samples (2 detik @ 250Hz)")
                        
                        # Preprocess - akan menghasilkan shape (1, 500)
                        features = preprocess_ekg(data)
            
                        # DEBUG: Tampilkan informasi
                        with st.expander("🔍 Detail Preprocessing", expanded=False):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Input data:** {len(data)} samples")
                                st.write(f"**Output shape:** {features.shape}")
                            with col2:
                                st.write(f"**Features range:** [{features.min():.3f}, {features.max():.3f}]")
                                st.write(f"**Features mean:** {features.mean():.3f}")
                            
                            # Plot window yang akan diprediksi
                            fig_debug = go.Figure()
                            fig_debug.add_trace(go.Scatter(
                                x=list(range(500)),
                                y=features[0],
                                mode='lines',
                                line=dict(color='green', width=2),
                                name='Window untuk Prediksi'
                            ))
                            fig_debug.update_layout(
                                title="Window 500 Samples yang akan Diprediksi",
                                xaxis_title="Sample Index (0-499)",
                                yaxis_title="Amplitude (Normalized)",
                                height=300
                            )
                            st.plotly_chart(fig_debug, use_container_width=True)
            
                        # PREDIKSI
                        with st.spinner("Melakukan prediksi..."):
                            prediction = model.predict(features)
                            prediction_proba = model.predict_proba(features)
            
                        # SIMPAN HASIL
                        st.session_state.prediction = {
                            'class': prediction[0],
                            'probability': prediction_proba[0],
                            'confidence': np.max(prediction_proba[0]),
                            'features_shape': features.shape
                        }
            
                        st.success("✅ Klasifikasi berhasil!")
            
                    except Exception as e:
                        st.error(f"❌ Error saat klasifikasi: {str(e)}")
                        # Tampilkan error detail untuk debugging
                        with st.expander("Detail Error"):
                            st.code(str(e))
                        st.session_state.prediction = None
