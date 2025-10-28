import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
import h5py
import yaml
import torch.nn.functional as F

from model.SelfSupervisedPretrainEMB import UnsupervisedPretrain
from utils import load_config

# === Colormap "hot" che mostra i NaN in bianco ===
cmap_hot_nan = cm.get_cmap('hot').copy()
cmap_hot_nan.set_bad(color='white')  # NaN = bianco


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_h5_sample(h5_path, index):
    with h5py.File(h5_path, "r") as f:
        data = f['signals'][index]  # [channels, time]
    return data


def visualize_masked_embedding(masked_emb, titolo):
    masked_emb_np = masked_emb.detach().cpu().numpy()

    if masked_emb_np.ndim == 3 and masked_emb_np.shape[0] == 1:
        masked_emb_np = masked_emb_np[0]

    binary_mask = (masked_emb_np == 0.0).astype(int)
    cmap = ListedColormap(["white", "blue"])

    plt.figure(figsize=(12, 6))
    plt.imshow(binary_mask, aspect='auto', cmap=cmap)
    plt.title(titolo)
    plt.xlabel("Dimensione dell'Embedding")
    plt.ylabel("Sequenza Temporale")
    plt.grid(True, color='gray', linewidth=0.5, linestyle='--')
    plt.xticks(np.arange(masked_emb_np.shape[1]))
    plt.yticks(np.arange(masked_emb_np.shape[0]))
    plt.tight_layout()
    plt.show()


def visualize_masked_diff(emb, pred_emb, mask, title="Errore di ricostruzione (emb - pred_emb)"):
    error = torch.abs(emb - pred_emb)  # [B, T, D]
    mask_np = mask.squeeze(0).detach().cpu().numpy()
    error_np = error.squeeze(0).detach().cpu().numpy()

    error_np_masked = np.where(mask_np == 1, error_np, np.nan)

    plt.figure(figsize=(12, 6))
    plt.imshow(error_np_masked, aspect='auto', cmap=cmap_hot_nan)
    plt.colorbar(label='Errore Assoluto')
    plt.title(title)
    plt.xlabel("Dimensione dell'Embedding")
    plt.ylabel("Sequenza Temporale")
    plt.tight_layout()
    plt.show()


def visualize_SE_Heatmap(emb, pred_emb, mask, title="Heatmap SE sulle posizioni mascherate"):
    mse = (emb - pred_emb) ** 2  # [B, T, D]
    mse_np = mse.squeeze(0).detach().cpu().numpy()
    mask_np = mask.squeeze(0).detach().cpu().numpy()

    mse_masked = np.where(mask_np == 1, mse_np, np.nan)

    plt.figure(figsize=(12, 6))
    plt.imshow(mse_masked, aspect='auto', cmap=cmap_hot_nan, interpolation='nearest')
    plt.colorbar(label='Errore Quadratico Medio')
    plt.title(title)
    plt.xlabel("Dimensione dell'Embedding")
    plt.ylabel("Sequenza Temporale")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config("configs/pretraining.yml")

    mean = np.load("mean.npy")  # shape (1, 19, 1)
    std = np.load("standard_deviation.npy")

    # === Carica un campione EEG ===
    sample = load_h5_sample("eeg_batch_001.h5", 0)  # [19, 1000]
    sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0).to(device)  # [1,19,1000]

    mean_t = torch.tensor(mean, dtype=torch.float32).to(device)
    std_t = torch.tensor(std, dtype=torch.float32).to(device)
  
    model = UnsupervisedPretrain(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_channels=config["n_channels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        mask_ratio=config["mask_ratio"]
    ).to(device)

    checkpoint_path = "/Users/danieleabbagnale/Desktop/EEGround/logs/pretrain_emb_mask_0.9/model_epoch_40.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sample_norm = (sample - mean_t) / std_t

    print("Esecuzione del modello...")
    with torch.no_grad():
        emb, mask, masked_emb, pred_emb = model(sample_norm)

    mse = F.mse_loss(pred_emb[mask], emb[mask])
    print(f"MSE (solo posizioni mascherate): {mse:.6f}")

    # === Visualizzazioni ===
    visualize_masked_embedding(masked_emb, "Embedding Mascherato")
    visualize_masked_embedding(pred_emb, "Embedding Ricostruito")
    visualize_masked_diff(emb, pred_emb, mask)
    visualize_SE_Heatmap(emb, pred_emb, mask)

    