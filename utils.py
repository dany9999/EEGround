
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

from torchmetrics import Metric
from sklearn.metrics import balanced_accuracy_score



class EEGDataset(Dataset):
    def __init__(self, h5_paths):
        self.file_paths = []
        self.idx_map = []  # (file_idx, segment_idx)
        self.h5_files = None  # sarà inizializzato nel worker

        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                n_segments = f["signals"].shape[0]
                self.file_paths.append(path)
                self.idx_map.extend([(len(self.file_paths) - 1, i) for i in range(n_segments)])

    def _init_files(self):
        self.h5_files = [h5py.File(path, 'r') for path in self.file_paths]

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        if self.h5_files is None:
            self._init_files()

        file_idx, seg_idx = self.idx_map[idx]
        f = self.h5_files[file_idx]
        data = f["signals"][seg_idx]
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor



def get_all_h5_files_from_tuh(base_path, sub_datasets):
    all_files = []
    for sub in sub_datasets:
        for label in ['Normal', 'Abnormal']:
            ref_path = os.path.join(base_path, sub, label, "REF")
            if os.path.exists(ref_path):
                files = [os.path.join(ref_path, f) for f in sorted(os.listdir(ref_path)) if f.endswith(".h5")]
                all_files.extend(files)
    return all_files

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