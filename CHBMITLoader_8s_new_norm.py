import os
import torch
import h5py
import numpy as np
import pandas as pd
from utils import load_config
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# =====================================================
# FUNZIONI DI NORMALIZZAZIONE (Z-SCORE)
# =====================================================

# def compute_global_channel_stats(loader, n_channels=16, clip=5.0):
#     """
#     Calcola media e dev_std globali per canale su tutti i segmenti del TRAIN loader.
#     Evita di caricare tutto in RAM (accumula streaming).
#     """
#     sum_c = np.zeros(n_channels)
#     sum_sq_c = np.zeros(n_channels)
#     count = 0

#     print("\n Calcolo statistiche globali per canale su TRAIN...")
#     for batch in loader:
#         x = batch["x"].numpy()  # [B, C, T]
#         sum_c += x.sum(axis=(0, 2))
#         sum_sq_c += (x ** 2).sum(axis=(0, 2))
#         count += x.shape[0] * x.shape[2]

#     mu = sum_c / count
#     sigma = np.sqrt(sum_sq_c / count - mu**2)
#     sigma = np.clip(sigma, 1e-6, np.inf)  # evita divisioni per zero

#     print("\n Statistiche globali calcolate:")
#     for i, (m, s) in enumerate(zip(mu, sigma)):
#         print(f"  Ch {i+1:02}: μ={m}, σ={s}")

#     return mu, sigma

def compute_global_channel(loader, n_channels=16):
    """
    Calcola media e deviazione standard per canale su tutto il TRAIN loader in modo incrementale.
    """
    print("\n Calcolo statistiche globali per canale (StandardScaler)...")

    # uno scaler per ogni canale
    scalers = [StandardScaler(copy=False) for _ in range(n_channels)]

    for batch_idx, batch in enumerate(loader):
        x = batch["x"].numpy()  # [B, C, T]
        B, C, T = x.shape

        # Appiattisci la dimensione (B*T) per ogni canale e aggiorna in streaming
        for c in range(C):
            data_c = x[:, c, :].reshape(-1, 1)
            scalers[c].partial_fit(data_c)

        if batch_idx % 10 == 0:
            print(f"  Elaborati {batch_idx+1}/{len(loader)} batch...")

    mu = np.array([sc.mean_[0] for sc in scalers])
    sigma = np.array([np.sqrt(sc.var_[0]) for sc in scalers])
    sigma = np.clip(sigma, 1e-6, np.inf)

    print("\n Statistiche globali calcolate:")
    for i, (m, s) in enumerate(zip(mu, sigma)):
        print(f"  Ch {i+1:02}: μ={m:.6e}, σ={s:.6e}")

    return mu, sigma


def apply_zscore(x, mu, sigma, clip=5.0):
    """Applica z-score per canale e clip a ±clip."""
    x = (x - mu[:, None]) / (sigma[:, None] + 1e-8)
    x = np.clip(x, -clip, clip)
    return x.astype(np.float32)


# =====================================================
# Dataset CHB-MIT
# =====================================================

class CHBMITAllSegmentsLabeledDataset(Dataset):
    def __init__(self, patient_ids, data_dir, gt_dir,
                 segment_duration_sec=8, transform=None,
                 mu=None, sigma=None):
        """
        Dataset CHB-MIT:
        - segmenti da 8s a 250 Hz
        - labeling secondo intervalli di crisi nel CSV GT
        - normalizzazione opzionale z-score per canale
        """
        self.index = []  # (fpath, seg_idx, label, file_id)
        self.transform = transform
        self.segment_duration_sec = segment_duration_sec
        self.mu = mu
        self.sigma = sigma

        for patient in patient_ids:
            patient_folder = os.path.join(data_dir, patient)
            gt_file = os.path.join(gt_dir, f"{patient}.csv")

            if not os.path.exists(gt_file):
                print(f" GT non trovato per {patient}, salto...")
                continue

            gt_df = pd.read_csv(gt_file, sep=';', engine='python')
            seizure_map = {}

            for _, row in gt_df.iterrows():
                edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
                if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
                    seizure_map[edf_base] = []
                else:
                    intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
                    seizure_map[edf_base] = intervals

            for fname in sorted(os.listdir(patient_folder)):
                if not fname.endswith(".h5"):
                    continue
                edf_base = fname.replace(".h5", "").replace("eeg_", "")
                fpath = os.path.join(patient_folder, fname)

                with h5py.File(fpath, 'r') as f:
                    n_segments = f['signals'].shape[0]


                intervals = seizure_map.get(edf_base, [])
                for i in range(n_segments):
                    seg_start = i * self.segment_duration_sec
                    seg_end = seg_start + self.segment_duration_sec
                    label = 0
                    for (st, en) in intervals:
                        if (seg_start >= st and seg_end <= en):
                            label = 1
                            break
                    self.index.append((fpath, i, label, edf_base))

    def parse_intervals(self, start_val, end_val):
        intervals = []
        if pd.isna(start_val) or pd.isna(end_val):
            return intervals
        start_parts = str(start_val).split(',')
        end_parts = str(end_val).split(',')
        for s_group, e_group in zip(start_parts, end_parts):
            starts = [float(x) for x in s_group.split('-') if x.strip() not in ["", "0"]]
            ends = [float(x) for x in e_group.split('-') if x.strip() not in ["", "0"]]
            for st, en in zip(starts, ends):
                intervals.append((st, en))
        return intervals

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, seg_idx, label, file_id = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['signals'][seg_idx][:16]  # (channels, time)
            #print("Shape:", x.shape)
            #print("Range:", x.min(), x.max())
            #print("Mean:", x.mean(), "Std:", x.std())

        if self.mu is not None and self.sigma is not None:
            x = apply_zscore(x, self.mu, self.sigma, clip=5.0)
               

        x = torch.tensor(x, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return {"x": x, "y": torch.tensor(label, dtype=torch.long)}


# =====================================================
# DataLoader helper
# =====================================================

def make_loader(patient_ids, dataset_path, gt_path, config,
                shuffle=True, balanced=False, neg_to_pos_ratio=5,
                mu=None, sigma=None):

    dataset = CHBMITAllSegmentsLabeledDataset(
        patient_ids=patient_ids,
        data_dir=dataset_path,
        gt_dir=gt_path,
        segment_duration_sec=config.get("segment_duration_sec", 8),
        transform=None,
        mu=mu,
        sigma=sigma
    )

    if balanced:
        labels = np.array([label for _, _, label, _ in dataset.index])
        pos_indices = np.where(labels == 1)[0]
        neg_indices = np.where(labels == 0)[0]

        num_pos = len(pos_indices)
        num_neg_to_keep = min(len(neg_indices), num_pos * neg_to_pos_ratio)

        np.random.seed(42)
        sampled_neg_indices = np.random.choice(neg_indices, size=num_neg_to_keep, replace=False)

        final_indices = np.concatenate([pos_indices, sampled_neg_indices])
        np.random.shuffle(final_indices)
        dataset.index = [dataset.index[i] for i in final_indices]

    loader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=shuffle,
                        num_workers=config["num_workers"],
                        pin_memory=False)
    return loader


# =====================================================
# MAIN – verifica dataset e normalizzazione
# =====================================================

if __name__ == "__main__":
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients   = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients  = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]

    config = load_config("configs/finetuning.yml")
    dataset_path = config["dataset_path_8s"]
    gt_path = "../../Datasets/chb_mit/GT"

    # Step 1️ - loader temporaneo (train completo)
    loader_tmp = make_loader(train_patients, dataset_path, gt_path, config,
                             shuffle=False, balanced=False)

    # Step 2️ - calcolo statistiche globali
    mu, sigma = compute_global_channel_stats(loader_tmp, n_channels=16)
    np.save("mu_train_finetuning_8s.npy", mu)
    np.save("sigma_train_finetuning_8s.npy", sigma)

    # Step 3️ - loader finali con z-score
    loader_train = make_loader(train_patients, dataset_path, gt_path, config,
                               shuffle=True, balanced=True, mu=mu, sigma=sigma)
    loader_val   = make_loader(val_patients, dataset_path, gt_path, config,
                               shuffle=False, mu=mu, sigma=sigma)
    loader_test  = make_loader(test_patients, dataset_path, gt_path, config,
                               shuffle=False, mu=mu, sigma=sigma)

    print(f"\nTrain set: {len(loader_train.dataset)} samples")
    print(f"Validation set: {len(loader_val.dataset)} samples")
    print(f"Test set: {len(loader_test.dataset)} samples")

    # Verifica forme e durata
    sample = loader_train.dataset[0]
    x, y = sample["x"], sample["y"]
    sec = x.shape[1] / 250
    print(f"\nSegmento esempio: shape={tuple(x.shape)}, label={y.item()}, durata={sec:.2f}s")