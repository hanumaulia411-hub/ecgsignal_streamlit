# preprocess.py - VERSI DENGAN DWT DENOISING
import numpy as np
from scipy import signal
import pywt

def dwt_denoise(signal_data, wavelet='db6', level=3):
    """
    DWT denoising sama persis seperti di Colab
    
    Parameters:
    -----------
    signal_data : array-like
        Sinyal ECG mentah
    wavelet : str
        Jenis wavelet (default: 'db6' seperti di Colab)
    level : int
        Level dekomposisi (default: 3 seperti di Colab)
    
    Returns:
    --------
    denoised_signal : numpy array
        Sinyal yang sudah di-denoising
    """
    try:
        # 1. Wavelet decomposition
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        # 2. Hitung threshold seperti di Colab
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = 0.3 * sigma * np.sqrt(2 * np.log(len(signal_data)))
        
        # 3. Apply threshold hanya ke detail coefficients tinggi (level >= 3)
        coeffs_thresh = [coeffs[0]]  # Approximation coefficient pertama
        
        for i, c in enumerate(coeffs[1:], start=1):
            if i >= 3:  # hanya detail frekuensi tinggi (sama seperti Colab)
                coeffs_thresh.append(pywt.threshold(c, uthresh, mode='soft'))
            else:
                coeffs_thresh.append(c)
        
        # 4. Rekonstruksi sinyal
        signal_denoised = pywt.waverec(coeffs_thresh, wavelet)
        
        # 5. Penyesuaian panjang (sama seperti Colab)
        return signal_denoised[:len(signal_data)]
        
    except Exception as e:
        print(f"[DWT ERROR] {str(e)}")
        return signal_data  # Return original jika error

def preprocess_ekg(ekg_signal, window_length=500, sampling_rate=250):
    """
    Preprocessing lengkap dengan DWT denoising seperti di Colab
    
    Flow: Upload -> DWT Denoise -> Window 500 samples -> Normalize -> Predict
    """
    try:
        # 1. Validasi input
        if ekg_signal is None or len(ekg_signal) == 0:
            print("[PREPROCESS] Input kosong")
            return np.zeros((1, window_length))
        
        # Konversi ke numpy array
        if not isinstance(ekg_signal, np.ndarray):
            ekg_signal = np.array(ekg_signal, dtype=np.float32)
        
        print(f"[PREPROCESS] Input: {len(ekg_signal)} samples")
        
        # 2. DWT DENOISING (sama seperti di Colab)
        if len(ekg_signal) >= 100:  # Minimal untuk DWT
            ekg_denoised = dwt_denoise(ekg_signal, wavelet='db6', level=3)
            print(f"[PREPROCESS] DWT denoising applied")
        else:
            ekg_denoised = ekg_signal
            print(f"[PREPROCESS] Skipping DWT (signal too short)")
        
        # 3. Pilih window untuk prediksi
        # Strategi: Ambil window pertama, atau cari window dengan variasi terbaik
        if len(ekg_denoised) >= window_length:
            # Pilih window dengan energi tertinggi (biasanya bagian yang informatif)
            # Atau ambil window dari tengah sinyal
            start_idx = len(ekg_denoised) // 2 - window_length // 2
            start_idx = max(0, start_idx)
            window_data = ekg_denoised[start_idx:start_idx + window_length]
        else:
            # Jika sinyal lebih pendek, pad dengan nol
            window_data = np.zeros(window_length)
            window_data[:len(ekg_denoised)] = ekg_denoised
            print(f"[PREPROCESS] Padding applied: {len(ekg_denoised)} -> {window_length}")
        
        # 4. Normalisasi window (sama seperti training di Colab)
        window_mean = np.mean(window_data)
        window_std = np.std(window_data)
        
        if window_std > 0:
            window_normalized = (window_data - window_mean) / window_std
        else:
            window_normalized = window_data - window_mean
        
        print(f"[PREPROCESS] Normalized - Mean: {np.mean(window_normalized):.4f}, Std: {np.std(window_normalized):.4f}")
        
        # 5. Reshape untuk model: (1, 500)
        features = window_normalized.reshape(1, -1)
        
        print(f"[PREPROCESS] Output shape: {features.shape}")
        
        return features
        
    except Exception as e:
        print(f"[PREPROCESS ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return np.zeros((1, window_length))
