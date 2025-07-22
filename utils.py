
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
from sklearn.metrics import balanced_accuracy_score


# ==== Utils ====

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
    def __init__(self):
        self.cache = {}

    def get_mean_std_for_file(self, file_path, device):
        file_path = os.path.abspath(file_path)
        folder = os.path.dirname(file_path)

        if folder not in self.cache:
            mean_path = os.path.join(folder, "mean.npy")
            std_path = os.path.join(folder, "standard_deviation.npy")

            mean_all = np.load(mean_path).squeeze()
            std_all = np.load(std_path).squeeze()

            self.cache[folder] = {
                "mean_all": mean_all,
                "std_all": std_all
            }

        mean = torch.tensor(self.cache[folder]["mean_all"], dtype=torch.float32).to(device)
        std = torch.tensor(self.cache[folder]["std_all"], dtype=torch.float32).to(device)
        return mean, std
    

# ==== Dataset ====

class EEGDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def convert_to_bipolar(data):
    """
    Converte segnali EEG in montaggio bipolare stile CHB-MIT.

    Args:
        data: np.ndarray con shape (batch, channels, samples)

    Returns:
        bipolar_data: np.ndarray con shape (batch, bipolar_channels, samples)
        bipolar_names: lista di stringhe con i nomi delle derivazioni
    """

    # Canali TUH, nell'ordine previsto nei dati
    tuh_channels = [
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6',
        'FZ', 'CZ', 'PZ'
    ]

    # Coppie per montaggio bipolare stile CHB-MIT
    bipolar_pairs = [
        ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
        ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
        ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
        ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
        ('FZ', 'CZ'), ('CZ', 'PZ')
    ]

    # Mappa nome → indice canale
    channel_indices = {}
    for idx, name in enumerate(tuh_channels):
        channel_indices[name] = idx

    # Costruzione dati bipolari
    bipolar_data = []
    bipolar_names = []

    for ch1, ch2 in bipolar_pairs:
        if ch1 not in channel_indices or ch2 not in channel_indices:
            print(f" Canali mancanti nel segnale: {ch1}-{ch2}")
            continue

        i1 = channel_indices[ch1]
        i2 = channel_indices[ch2]

        signal1 = data[:, i1, :]  # (batch, samples)
        signal2 = data[:, i2, :]
        diff = signal1 - signal2  # differenza bipolare

        bipolar_data.append(diff)
        bipolar_names.append(f"{ch1}-{ch2}")

    # Stack lungo l'asse dei canali → (batch, bipolar_channels, samples)
    bipolar_numpy = np.stack(bipolar_data, axis=1)

    return bipolar_numpy, bipolar_names



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



class BinaryBalancedAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("preds", default=torch.tensor([], dtype=torch.float32), dist_reduce_fx="cat")
        self.add_state("targets", default=torch.tensor([], dtype=torch.long), dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # Assicura che siano 1D
        preds = preds.detach().view(-1)
        targets = targets.detach().view(-1)

        # Converti i target a long (interi binari)
        targets = (targets > 0.5).long()

        # Salva
        self.preds = torch.cat([self.preds, preds])
        self.targets = torch.cat([self.targets, targets])

    def compute(self):
        print("preds_np.shape:", self.preds_np.shape)
        print("targets_np.shape:", self.targets_np.shape)
        preds_np = self.preds.cpu().numpy().reshape(-1)      
        targets_np = self.targets.cpu().numpy().reshape(-1)  

        binary_preds = (preds_np >= 0.5).astype(int)

        return balanced_accuracy_score(targets_np, binary_preds)

    def reset(self):
        super().reset()  
       
