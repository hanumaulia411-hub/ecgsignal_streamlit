# app.py - VERSI DIPERBAIKI UNTUK STREAMLIT CLOUD
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from preprocess import preprocess_ekg, dwt_denoise

# ========== KONFIGURASI HALAMAN ==========
st.set_page_config(
    page_title="Klasifikasi ECG - Aktif vs Tenang",
    layout="wide",
    initial_sidebar_state="expanded"  # Sidebar terbuka
)

# ========== INISIALISASI SESSION STATE ==========
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# ========== JUDUL UTAMA ==========
st.title("Klasifikasi Sinyal ECG: Aktif vs Tenang")
st.markdown("Upload sinyal ECG dalam format CSV untuk diklasifikasikan")
st.divider()

# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf_model.pkl')
        st.success("✅ Model berhasil dimuat")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ========== FUNGSI BANTU ==========
def read_uploaded_file(file):
    """Membaca file CSV yang diupload"""
    try:
        if file is None:
            return None
        
        # Coba beberapa format CSV
        try:
            # Format 1: Satu kolom tanpa header
            df = pd.read_csv(file, header=None)
        except:
            # Format 2: Dengan header
            file.seek(0)  # Reset file pointer
            df = pd.read_csv(file)
        
        # Ambil kolom pertama (asumsi sinyal ada di kolom pertama)
        if df.shape[1] > 0:
            signal = df.iloc[:, 0].values
            signal = signal.astype(np.float32)
            
            # Hapus NaN jika ada
            signal = signal[~np.isnan(signal)]
            
            return signal
        else:
            st.error("File tidak berisi data")
            return None
            
    except Exception as e:
        st.error(f"Gagal membaca file: {str(e)}")
        return None

# ========== SIDEBAR ==========
st.sidebar.header("📁 **Upload Data ECG**")
st.sidebar.markdown("---")

# File uploader di sidebar
uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV",
    type=['csv', 'txt'],
    help="File harus berisi sinyal ECG (satu kolom)"
)

# Informasi di sidebar
st.sidebar.markdown("---")
st.sidebar.info("""
**📋 Format File:**
- CSV dengan satu kolom
- Nilai numerik (float)
- Minimal 500 sample
- Contoh: `0.952, 1.023, 0.876, ...`
""")

# Contoh data download
st.sidebar.markdown("---")
st.sidebar.subheader("📥 Contoh Data")

if st.sidebar.button("Generate Contoh Data"):
    # Generate contoh sinyal ECG
    t = np.linspace(0, 8*np.pi, 2000)
    example_signal = np.sin(t) + 0.3*np.sin(3*t) + 0.1*np.random.randn(2000)
    
    # Buat DataFrame dan simpan
    df_example = pd.DataFrame(example_signal, columns=['ECG_Signal'])
    csv = df_example.to_csv(index=False)
    
    # Download button
    st.sidebar.download_button(
        label="⬇️ Download Contoh CSV",
        data=csv,
        file_name="contoh_ecg.csv",
        mime="text/csv"
    )

# ========== MAIN APP ==========
def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Aplikasi tidak dapat berjalan tanpa model. Pastikan file rf_model.pkl ada.")
        st.stop()
    
    # ========== TAMPILAN UTAMA BERDASARKAN STATE ==========
    
    # STATE 1: Belum ada file yang diupload
    if uploaded_file is None and st.session_state.uploaded_data is None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("**Silakan upload file CSV melalui sidebar di sebelah kiri**")
            
            st.markdown("""
            ### 📝 Petunjuk Penggunaan:
            1. **Upload file** CSV melalui sidebar
            2. **Pratinjau sinyal** akan muncul otomatis
            3. **Klik 'Mulai Klasifikasi'** untuk analisis
            4. **Lihat hasil** dan interpretasi
            
            ### 🔧 Spesifikasi Teknis:
            - Model: Random Forest (500 fitur)
            - Preprocessing: DWT denoising (db6, level=3)
            - Window: 500 samples (2 detik @ 250Hz)
            - Output: Aktif atau Tenang
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/300/000000/ecg.png", 
                    caption="ECG Signal Analysis")
            
            st.markdown("""
            **📊 Contoh Format:**
            ```
            0.952
            1.023
            0.876
            -0.234
            0.567
            ```
            """)
        
        return
    
    # STATE 2: Ada file baru yang diupload
    if uploaded_file is not None and st.session_state.file_name != uploaded_file.name:
        with st.spinner("Membaca file..."):
            data = read_uploaded_file(uploaded_file)
            
            if data is not None:
                st.session_state.uploaded_data = data
                st.session_state.file_name = uploaded_file.name
                st.session_state.prediction = None  # Reset prediksi lama
                
                st.success(f"✅ File '{uploaded_file.name}' berhasil dibaca")
                st.rerun()  # Refresh untuk tampilkan data
            else:
                st.error("Gagal membaca file")
                return
    
    # STATE 3: Data sudah diupload, tampilkan
    if st.session_state.uploaded_data is not None:
        data = st.session_state.uploaded_data
        
        # Header dengan nama file
        st.subheader(f"📄 File: {st.session_state.file_name}")
        
        # ========== TAB UNTUK VISUALISASI ==========
        tab1, tab2, tab3 = st.tabs(["📈 Raw Signal", "🌀 After DWT", "🎯 Prediction Window"])
        
        with tab1:
            st.markdown("### Sinyal ECG Mentah")
            
            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
            with col_stat1:
                st.metric("Total Samples", f"{len(data):,}")
            with col_stat2:
                st.metric("Mean", f"{np.mean(data):.4f}")
            with col_stat3:
                st.metric("Std Dev", f"{np.std(data):.4f}")
            with col_stat4:
                st.metric("Duration", f"{len(data)/250:.1f} detik")
            
            # Plot sinyal mentah
            display_points = min(1000, len(data))
            fig_raw = go.Figure()
            fig_raw.add_trace(go.Scatter(
                x=list(range(display_points)),
                y=data[:display_points],
                mode='lines',
                line=dict(color='blue', width=1),
                name='Raw ECG'
            ))
            fig_raw.update_layout(
                title=f"Raw ECG Signal (First {display_points} samples)",
                xaxis_title="Sample Index",
                yaxis_title="Amplitude",
                height=400
            )
            st.plotly_chart(fig_raw, use_container_width=True)
        
        with tab2:
            st.markdown("### Setelah DWT Denoising (db6, level=3)")
            
            # Apply DWT
            try:
                data_dwt = dwt_denoise(data)
                
                col_dwt1, col_dwt2, col_dwt3 = st.columns(3)
                with col_dwt1:
                    noise_reduction = (1 - np.std(data_dwt)/np.std(data)) * 100
                    st.metric("Noise Reduction", f"{noise_reduction:.1f}%")
                with col_dwt2:
                    st.metric("New Mean", f"{np.mean(data_dwt):.4f}")
                with col_dwt3:
                    st.metric("New Std", f"{np.std(data_dwt):.4f}")
                
                # Plot setelah DWT
                fig_dwt = go.Figure()
                fig_dwt.add_trace(go.Scatter(
                    x=list(range(min(1000, len(data_dwt)))),
                    y=data_dwt[:min(1000, len(data_dwt))],
                    mode='lines',
                    line=dict(color='green', width=1.5),
                    name='DWT Denoised'
                ))
                fig_dwt.update_layout(
                    title="After Wavelet Denoising",
                    xaxis_title="Sample Index",
                    yaxis_title="Amplitude",
                    height=400
                )
                st.plotly_chart(fig_dwt, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error dalam DWT processing: {str(e)}")
        
        with tab3:
            st.markdown("### Window 500 Samples untuk Prediksi")
            
            # Preprocess untuk dapat window
            features = preprocess_ekg(data)
            
            st.info(f"**Window Shape:** {features.shape} | **Sample Rate:** 250Hz | **Duration:** 2 detik")
            
            # Plot window
            fig_window = go.Figure()
            fig_window.add_trace(go.Scatter(
                x=list(range(500)),
                y=features[0],
                mode='lines',
                line=dict(color='red', width=2),
                name='Prediction Window'
            ))
            fig_window.update_layout(
                title="500-Sample Window (Normalized)",
                xaxis_title="Sample Index (0-499)",
                yaxis_title="Normalized Amplitude",
                height=400
            )
            st.plotly_chart(fig_window, use_container_width=True)
            
            # Statistik window
            st.write("**Window Statistics:**")
            col_win1, col_win2, col_win3, col_win4 = st.columns(4)
            with col_win1:
                st.metric("Window Mean", f"{np.mean(features[0]):.4f}")
            with col_win2:
                st.metric("Window Std", f"{np.std(features[0]):.4f}")
            with col_win3:
                st.metric("Min", f"{np.min(features[0]):.4f}")
            with col_win4:
                st.metric("Max", f"{np.max(features[0]):.4f}")
        
        # ========== TOMBOL KLASIFIKASI ==========
        st.divider()
        st.subheader("🔬 Analisis Klasifikasi")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
        
        with col_btn1:
            classify_btn = st.button("🚀 **Mulai Klasifikasi**", 
                                    type="primary", 
                                    use_container_width=True)
        
        with col_btn2:
            if st.button("🔄 **Upload File Baru**", use_container_width=True):
                # Reset state
                st.session_state.uploaded_data = None
                st.session_state.file_name = None
                st.session_state.prediction = None
                st.rerun()
        
        # Jika tombol klasifikasi ditekan
        if classify_btn:
            with st.spinner("Melakukan klasifikasi..."):
                try:
                    # Predict
                    prediction = model.predict(features)
                    prediction_proba = model.predict_proba(features)
                    
                    # Simpan hasil
                    st.session_state.prediction = {
                        'class': prediction[0],
                        'probability': prediction_proba[0],
                        'confidence': np.max(prediction_proba[0]),
                        'class_name': "AKTIF" if prediction[0] == 1 else "TENANG"
                    }
                    
                    st.success("✅ Klasifikasi berhasil!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error dalam klasifikasi: {str(e)}")
        
        # ========== TAMPILKAN HASIL JIKA ADA ==========
        if st.session_state.prediction is not None:
            st.divider()
            st.subheader("🎯 **Hasil Klasifikasi**")
            
            pred = st.session_state.prediction
            confidence_percent = pred['confidence'] * 100
            
            # Tampilan hasil utama
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                # Box dengan warna berdasarkan kelas
                if pred['class_name'] == "AKTIF":
                    st.markdown("""
                    <div style='background-color: #ff6b6b; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>AKTIF</h2>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style='background-color: #51cf66; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h2 style='color: white; margin: 0;'>TENANG</h2>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_res2:
                st.metric("Confidence Level", f"{confidence_percent:.1f}%")
                st.progress(pred['confidence'])
            
            with col_res3:
                # Probabilitas detail
                prob_df = pd.DataFrame({
                    'Kelas': ['Tenang', 'Aktif'],
                    'Probabilitas': (pred['probability'] * 100).round(1)
                })
                st.dataframe(prob_df, use_container_width=True)
            
            # Interpretasi
            st.subheader("💡 **Interpretasi Medis**")
            
            if pred['class_name'] == "AKTIF":
                st.warning("""
                **🫀 Aktivitas Jantung Tinggi Terdeteksi**
                
                **Kemungkinan Kondisi:**
                - Aktivitas fisik sedang/berat
                - Stres atau kecemasan
                - Konsumsi kafein/stimulan
                - Demam atau infeksi
                - Hipertiroidisme
                
                **Rekomendasi:**
                1. Monitor denyut jantung
                2. Istirahat jika sedang beraktivitas
                3. Konsultasi dokter jika:
                   - Disertai nyeri dada
                   - Pusing atau sesak napas
                   - Berlangsung terus-menerus
                """)
            else:
                st.success("""
                **🫀 Aktivitas Jantung Normal/Tenang Terdeteksi**
                
                **Kemungkinan Kondisi:**
                - Istirahat atau tidur
                - Kondisi relaksasi
                - Denyut jantung basal normal
                - Kondisi meditasi
                
                **Rekomendasi:**
                1. Kondisi normal
                2. Lanjutkan aktivitas rutin
                3. Monitor jika ada gejala lain
                """)

# ========== JALANKAN APLIKASI ==========
if __name__ == "__main__":
    main()
