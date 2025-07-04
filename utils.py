
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



class CHBMITLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # 2560 -> 2000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        

        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config




class BinaryBalancedAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        self.preds.append(preds.detach().cpu())
        self.targets.append(target.detach().cpu())

    def compute(self):
        preds = torch.cat(self.preds).numpy()
        targets = torch.cat(self.targets).numpy()
        return balanced_accuracy_score(targets, preds >= 0.5)  # soglia binaria

    def reset(self):
        self.preds.clear()
        self.targets.clear()