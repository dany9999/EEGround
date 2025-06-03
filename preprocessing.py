


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import torch

# --- Funzioni di preprocessing ---

def z_score_normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def percentile_95_normalize(x):
    """
    Normalizza ogni campione nel batch dividendo per il 95° percentile.
    Assunzione: x è un tensore di forma [B, C, T]
    """
    # Calcola il 95° percentile lungo l'asse del tempo (T)
    p95 = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True)
    
    # Evita la divisione per zero
    p95[p95 == 0] = 1e-6

    return x / p95

def bandpass_filter(signal, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# --- Caricamento di un segmento di esempio ---
# Supponiamo: (channels, 2560), prendiamo il primo canale
def main():
    
    segments = np.load("./CHB-MIT/processed_numpy/all_segments.npy")
    sample = segments[0][0]  # primo segmento, primo canale

    fs = 256  # frequenza di campionamento
    t = np.arange(sample.shape[0]) / fs  # asse temporale in secondi

    # --- Visualizzazione step-by-step ---
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # 1. Segnale originale
    axs[0].plot(t, sample, color='black')
    axs[0].set_title("Segnale Originale (raw)")

    # 2. Z-score Normalization
    zscored = z_score_normalize(sample)
    axs[1].plot(t, zscored, color='blue')
    axs[1].set_title("Z-score Normalization")

    # 3. Normalizzazione al 95° percentile
    percentiled = percentile_95_normalize(sample)
    axs[2].plot(t, percentiled, color='green')
    axs[2].set_title("Normalizzazione al 95° Percentile")

    # 4. Filtro passa-banda (0.5 – 40 Hz)
    filtered = bandpass_filter(sample, 0.5, 40, fs)
    axs[3].plot(t, filtered, color='purple')
    axs[3].set_title("Filtro Passa Banda (0.5 – 40 Hz)")

    # --- Layout finale ---
    for ax in axs:
        ax.set_ylabel("Ampiezza (uV)")
    axs[-1].set_xlabel("Tempo (s)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()