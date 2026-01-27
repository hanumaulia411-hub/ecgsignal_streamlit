# preprocess.py
import numpy as np
from scipy import signal

def preprocess_ekg(ekg_signal):
    """
    Preprocessing dan ekstraksi fitur dasar untuk sinyal EKG.
    Fungsi ini harus mengembalikan array fitur 2D dengan shape (1, n_features)
    agar sesuai dengan input model Random Forest (rf_model.pkl).
    """
    # 1. Pastikan input adalah numpy array
    if not isinstance(ekg_signal, np.ndarray):
        ekg_signal = np.array(ekg_signal)
    
    # 2. (Opsional) Normalisasi sederhana
    # ekg_normalized = (ekg_signal - np.mean(ekg_signal)) / np.std(ekg_signal)
    
    # 3. EKSTRAKSI FITUR SEDERHANA - ini yang menjadi variabel 'features'
    # Contoh: ekstrak beberapa fitur statistik dasar
    # Sesuaikan jumlah dan jenis fitur ini dengan apa yang digunakan saat training di Colab!
    features = np.array([
        np.mean(ekg_signal),        # Rata-rata
        np.std(ekg_signal),         # Simpangan baku
        np.min(ekg_signal),         # Nilai minimum
        np.max(ekg_signal),         # Nilai maksimum
        np.median(ekg_signal),      # Median
        np.mean(np.abs(ekg_signal - np.mean(ekg_signal)))  # Deviasi absolut rata-rata
    ])
    
    # 4. Reshape menjadi (1, n_features) karena model mengharapkan input 2D
    # Misal: dari array dengan 6 fitur -> shape (1, 6)
    features = features.reshape(1, -1)
    
    # Sekarang variabel 'features' sudah terdefinisi dan bisa di-return
    return features
