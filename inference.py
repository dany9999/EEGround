import os
import torch
import numpy as np
import pyedflib
import matplotlib.pyplot as plt
from model.SelfSupervisedPretrain import UnsupervisedPretrain
from utils import load_config

# # ====== Parametri ======
# EDF_PATH = "/Users/danieleabbagnale/Desktop/EEGround/CHB-MIT/Raw/chb01/chb01_01.edf"
# CHECKPOINT_PATH = "/Users/danieleabbagnale/Desktop/EEGround/logs/pretrain/model_epoch_6.pt"  # cambia se necessario
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# NUM_CHANNELS = 19
# SIGNAL_LEN = 1000

# # ====== Caricamento modello ======
# def load_model(checkpoint_path):
#     checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
#     config = checkpoint.get("config", {
#         "emb_size": 256,
#         "heads": 8,
#         "depth": 4,
#         "n_channels": NUM_CHANNELS,
#         "mask_ratio": 0.25
#     })

#     model = UnsupervisedPretrain(
#         emb_size=config["emb_size"],
#         heads=config["heads"],
#         depth=config["depth"],
#         n_channels=config["n_channels"],
#         mask_ratio=config["mask_ratio"]
#     ).to(DEVICE)

#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.eval()
#     return model

# # ====== Caricamento segnale da EDF ======
# def read_edf_select_channels_random(edf_path, num_channels=19, signal_len=1000):
#     f = pyedflib.EdfReader(edf_path)
#     all_channels = f.getSignalLabels()
#     total_signals = f.signals_in_file

#     if total_signals < num_channels:
#         raise ValueError(f"File has only {total_signals} channels, need at least {num_channels}.")

#     selected_indices = np.random.choice(total_signals, num_channels, replace=False)
#     selected_indices.sort()
#     selected_channels = [all_channels[i] for i in selected_indices]
#     print(f"Selected channels: {selected_channels}")

#     signal_array = np.stack([f.readSignal(i)[:signal_len] for i in selected_indices])
#     f._close()

#     return signal_array, selected_channels

# ====== Visualizzazione ======
def plot_signals(original, reconstructed, selected_channels, signal_len=1000):
    plt.figure(figsize=(15, 2 * len(original)))
    for i in range(len(original)):
        plt.subplot(len(original), 1, i+1)
        plt.plot(original[i], label="Original", color="blue", linewidth=1)
        plt.plot(reconstructed[i], label="Reconstructed", color="red", linestyle="--", linewidth=1)
        plt.title(selected_channels[i])
        plt.tight_layout()
    plt.legend()
    plt.show()

# # ====== Inference ======
# def run_inference():
#     print("Caricamento modello...")
#     model = load_model(CHECKPOINT_PATH)

#     print("Lettura EDF...")
#     signal_array, selected_channels = read_edf_select_channels_random(EDF_PATH, NUM_CHANNELS, SIGNAL_LEN)

#     signal_tensor = torch.tensor(signal_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # shape (1, C, T)

#     # Normalizzazione (z-score)
#     mean = signal_tensor.mean(dim=2, keepdim=True)
#     std = signal_tensor.std(dim=2, keepdim=True) + 1e-6
#     normalized = (signal_tensor - mean) / std

#     print("Esecuzione del modello...")
#     with torch.no_grad():
#         reconstructed = model(normalized)
 

#     # Calcolo MSE
#     mse = torch.mean((reconstructed - signal_tensor) ** 2).item()
#     print(f"MSE: {mse:.6f}")

#     # Visualizza
#     original_np = signal_tensor.squeeze(0).cpu().numpy()
#     reconstructed_np = reconstructed.squeeze(0).cpu().numpy()
#     plot_signals(original_np, reconstructed_np, selected_channels)

# if __name__ == "__main__":
#     run_inference()


import h5py
import numpy as np
import torch
import yaml
import torch.nn.functional as F

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_h5_sample(h5_path, index):
    with h5py.File(h5_path, "r") as f:
        data = f['signals'][index]  # [channels, time]
    return data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = load_config("configs/pretraining.yml")

    mean = np.load("mean.npy")          # shape (1, 19, 1)
    std = np.load("standard_deviation.npy")

    sample = load_h5_sample("eeg_batch_000.h5", 0)  # [19, 1000]

    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)  # [1,19,1000]

    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).to(device)

    

    model = UnsupervisedPretrain(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_channels=config["n_channels"],
        mask_ratio=config["mask_ratio"],
        signal_len=1000,
        n_fft=200,
        hop_length=100,
    ).to(device)

    checkpoint = torch.load("/Users/danieleabbagnale/Desktop/EEGround/logs/pretrain/model_epoch_2.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_norm = (sample - mean_t) / std_t



    print("Esecuzione del modello...")
    with torch.no_grad():
        reconstructed = model(sample_norm)
 

    # Calcolo MSE
    mse = F.mse_loss(reconstructed, sample)
    
    print(f"MSE: {mse:.6f}")

    # Visualizza
    original_np = sample.squeeze(0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(0).cpu().numpy()


    plt.figure(figsize=(15, 2 * original_np.shape[0]))
    for i in range(original_np.shape[0]):
        plt.subplot(original_np.shape[0], 1, i+1)
        plt.plot(original_np[i], label="Original", color="blue", linewidth=1)
        plt.plot(reconstructed_np[i], label="Reconstructed", color="red", linestyle="--", linewidth=1)
        plt.title(f"Channel {i+1}")
        plt.legend()
    plt.tight_layout()
    plt.show()

    print("Shape del segnale ricostruito:", reconstructed.shape)
