import numpy as np
from scipy import signal


def apply_filters(X: np.ndarray, fs: int = 300):
    """Apply 50 Hz notch filtering followed by a 0.5-45 Hz Butterworth bandpass."""
    print("Sinyal İşleme: Bandpass (0.5-45 Hz) ve Notch (50 Hz) filtreleri uygulanıyor...")
    X_filtered = np.zeros_like(X, dtype=np.float64)

    b_notch, a_notch = signal.iirnotch(w0=50.0, Q=30.0, fs=fs)
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    b_band, a_band = signal.butter(N=4, Wn=[low, high], btype="band")

    for i in range(X.shape[0]):
        temp_sig = signal.filtfilt(b_notch, a_notch, X[i])
        X_filtered[i] = signal.filtfilt(b_band, a_band, temp_sig)

    return X_filtered


def apply_zscore_normalization(X: np.ndarray):
    print("Veriler Z-Score ile normalize ediliyor...")
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    return (X - mean) / (std + 1e-8)
