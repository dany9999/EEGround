
from scipy.signal import resample
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
import pickle
import os

import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from glob import glob
from torchmetrics import Metric



# ==== Utils ====


# define focal loss on binary classification
def focal_loss(y_hat, y, alpha=0.8, gamma=0.7):
    # y_hat: (N, 1)
    # y: (N, 1)
    # alpha: float
    # gamma: float
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    # y_hat = torch.clamp(y_hat, -75, 75)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def collect_h5_files(root_dir):
    all_files = []
    subdatasets = ['TUAB', 'TUEP', 'TUEV', 'TUSZ']
    for sub in subdatasets:
        sub_path = os.path.join(root_dir, sub)
        if not os.path.exists(sub_path):
            continue
        for condition in ['Normal', 'Abnormal']:
            cond_path = os.path.join(sub_path, condition, 'REF')
            if os.path.exists(cond_path):
                files = glob(os.path.join(cond_path, "*.h5"))
                files = [f for f in files if not f.endswith(('mean.npy', 'standard_deviation.npy'))]
                all_files.extend(files)
    return sorted(all_files)




# ==== Mean/Std Loader ====

class MeanStdLoader:
    def __init__(self, global_mean, global_std, device):
        self.mean = torch.tensor(global_mean, dtype=torch.float32).to(device)
        self.std  = torch.tensor(global_std, dtype=torch.float32).to(device)

    def get_mean_std(self):
        return self.mean, self.std
    

# ==== Dataset ====

# class EEGDataset(Dataset):
#     def __init__(self, data_array):
#         self.data = torch.tensor(data_array, dtype=torch.float32)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]
    
class EEGDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]  # shape [C, T]

        # --- Normalizzazione per-segmento come BIOT ---
        p95 = torch.quantile(x.abs(), 0.95, dim=1, keepdim=True)  # shape [C, 1]
        p95 = torch.clamp(p95, min=1e-8)
        x = x / p95

        return x




def visualize_masked_embedding(self, masked_emb, titolo):
    """
    Visualizza una griglia binaria dell'embedding mascherato:
    celle blu per valori zero, bianche altrimenti.
    """
    
    # Converti in numpy
    masked_emb_np = masked_emb.detach().cpu().numpy()

    # Rimuovi la dimensione di batch se presente (es. da [1, T, D] a [T, D])
    if masked_emb_np.ndim == 3 and masked_emb_np.shape[0] == 1:
        masked_emb_np = masked_emb_np[0]

    # Crea maschera binaria: 1 se valore è zero, 0 altrimenti
    binary_mask = (masked_emb_np == 0.0).astype(int)

    # Colori: 0 → bianco, 1 → blu
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


       
def compute_global_stats(patient_ids, data_dir):
    """
    Calcola mean e std globali su tutti i segnali del train set.
    Ritorna tensori shape (channels, 1) in float32 per il broadcasting.
    """
    sum_vals = None
    sum_sq_vals = None
    total_samples = 0

    for patient in patient_ids:
        patient_folder = os.path.join(data_dir, patient)
        for fname in sorted(os.listdir(patient_folder)):
            if not fname.endswith(".h5"):
                continue
            fpath = os.path.join(patient_folder, fname)
            with h5py.File(fpath, "r") as f:
                signals = f["signals"][:, :18, :]  # (n_segments, ch, time)
                n, c, t = signals.shape
                reshaped = signals.transpose(1, 0, 2).reshape(c, -1).astype(np.float64)

                if sum_vals is None:
                    sum_vals = reshaped.sum(axis=1)
                    sum_sq_vals = (reshaped ** 2).sum(axis=1)
                else:
                    sum_vals += reshaped.sum(axis=1)
                    sum_sq_vals += (reshaped ** 2).sum(axis=1)

                total_samples += reshaped.shape[1]

    if sum_vals is None:
        raise RuntimeError("Nessun file valido trovato per calcolare mean/std")

    mean = sum_vals / total_samples
    var = (sum_sq_vals / total_samples) - (mean ** 2)
    std = np.sqrt(var) + 1e-6

    return mean.reshape(-1, 1).astype(np.float32), std.reshape(-1, 1).astype(np.float32)