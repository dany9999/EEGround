import os
import torch
import h5py
import numpy as np
import pandas as pd
from utils import load_config
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import resample_poly  # per il downsampling


# class CHBMITAllSegmentsLabeledDataset(Dataset):
#     def __init__(self, patient_ids, data_dir, gt_dir,
#                  segment_duration_sec=4, transform=None):
#         """
#         Crea finestre da 8s unendo due segmenti consecutivi da 4s (250 Hz),
#         effettua resampling a 200 Hz,
#         e aggiunge campioni extra con overlap 4s nei periodi di crisi.
#         """
#         self.index = []  # (fpath, seg_idx, label, file_id)
#         self.transform = transform
#         self.segment_duration_sec = segment_duration_sec

#         for patient in patient_ids:
#             patient_folder = os.path.join(data_dir, patient)
#             gt_file = os.path.join(gt_dir, f"{patient}.csv")
#             if not os.path.exists(gt_file):
#                 print(f" GT not found for {patient}, skipping")
#                 continue

#             gt_df = pd.read_csv(gt_file, sep=';', engine='python')
#             seizure_map = {}
#             for _, row in gt_df.iterrows():
#                 edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
#                 if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
#                     seizure_map[edf_base] = []
#                 else:
#                     intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
#                     seizure_map[edf_base] = intervals

#             # Scansiona gli .h5 del paziente
#             for fname in sorted(os.listdir(patient_folder)):
#                 if not fname.endswith(".h5"):
#                     continue
#                 edf_base = fname.replace(".h5", "").replace("eeg_", "")
#                 fpath = os.path.join(patient_folder, fname)

#                 with h5py.File(fpath, 'r') as f:
#                     n_segments = f['signals'].shape[0]

#                 intervals = seizure_map.get(edf_base, [])

#                 # === Segmenti base (8 s, non sovrapposti) ===
#                 for i in range(0, n_segments - 1, 2):  # passo di 2 segmenti da 4s
#                     seg_start = i * self.segment_duration_sec
#                     seg_end = seg_start + 2 * self.segment_duration_sec
#                     label = 0
#                     for (st, en) in intervals:
#                         if (seg_start >= st and seg_start < en) or (seg_end > st and seg_end <= en):
#                             label = 1
#                             break
#                     self.index.append((fpath, i, label, edf_base))

#                 # === Campioni aggiuntivi (solo dentro intervalli epilettici, overlap 4 s) ===
#                 if len(intervals) > 0:
#                     for (st, en) in intervals:
#                         # converti tempi in indici di segmento (4 s ciascuno)
#                         start_idx = int(st // self.segment_duration_sec)
#                         end_idx = int(en // self.segment_duration_sec)
#                         # scorri a passo 1 → overlap 4 s (cioè 50%)
#                         for i in range(start_idx, max(start_idx + 1, end_idx - 1), 1):
#                             if i + 1 >= n_segments:
#                                 break
#                             self.index.append((fpath, i, 1, edf_base))  # label forzato = 1

#         # Rimuovi eventuali duplicati (stesso file e indice)
#         self.index = list(dict.fromkeys(self.index))

#     def parse_intervals(self, start_val, end_val):
#         intervals = []
#         if pd.isna(start_val) or pd.isna(end_val):
#             return intervals
#         start_parts = str(start_val).split(',')
#         end_parts = str(end_val).split(',')
#         for s_group, e_group in zip(start_parts, end_parts):
#             starts = [float(x) for x in s_group.split('-') if x.strip() not in ["", "0"]]
#             ends = [float(x) for x in e_group.split('-') if x.strip() not in ["", "0"]]
#             for st, en in zip(starts, ends):
#                 intervals.append((st, en))
#         return intervals

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         fpath, seg_idx, label, file_id = self.index[idx]
#         with h5py.File(fpath, 'r') as f:
#             signals = f['signals']
#             x1 = signals[seg_idx][:16]
#             x2 = signals[seg_idx + 1][:16]
#             x_8s = np.concatenate([x1, x2], axis=-1)

#         # Downsampling 250 → 200 Hz
#         x_200 = resample_poly(x_8s, up=4, down=5, axis=-1)

#         # Normalizzazione percentile 95
#         x_200 = x_200 / (np.quantile(np.abs(x_200), q=0.95, axis=-1, keepdims=True) + 1e-8)
#         x_200 = torch.tensor(x_200, dtype=torch.float32)
#         if self.transform:
#             x_200 = self.transform(x_200)

#         return {"x": x_200, "y": torch.tensor(label, dtype=torch.long)}
    

class CHBMITAllSegmentsLabeledDataset(Dataset):
    def __init__(self, patient_ids, data_dir, gt_dir, segment_duration_sec=4, transform=None):
        self.transform = transform
        self.segment_duration_sec = segment_duration_sec
        self.index = []
        self.cache = {}  # memorizza segnali in RAM

        for patient in patient_ids:
            patient_folder = os.path.join(data_dir, patient)
            gt_file = os.path.join(gt_dir, f"{patient}.csv")
            if not os.path.exists(gt_file):
                continue

            gt_df = pd.read_csv(gt_file, sep=";", engine="python")
            seizure_map = {}
            for _, row in gt_df.iterrows():
                edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
                if "no seizure" in str(row["class_name"]).lower():
                    seizure_map[edf_base] = []
                else:
                    intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
                    seizure_map[edf_base] = intervals

            for fname in sorted(os.listdir(patient_folder)):
                if not fname.endswith(".h5"):
                    continue
                edf_base = fname.replace(".h5", "").replace("eeg_", "")
                fpath = os.path.join(patient_folder, fname)

                #  Carica e conserva in memoria
                with h5py.File(fpath, "r") as f:
                    self.cache[fpath] = f["signals"][:]  # numpy array completo

                n_segments = self.cache[fpath].shape[0]
                intervals = seizure_map.get(edf_base, [])
                for i in range(0, n_segments - 1, 2):
                    seg_start = i * self.segment_duration_sec
                    seg_end = seg_start + 2 * self.segment_duration_sec
                    label = 0
                    for (st, en) in intervals:
                        if (seg_start >= st and seg_start < en) or (seg_end > st and seg_end <= en):
                            label = 1
                            break
                    self.index.append((fpath, i, label, edf_base))

        print(f"Cache caricata con {len(self.cache)} file HDF5")

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
        fpath, seg_idx, label, _ = self.index[idx]
        signals = self.cache[fpath]  # 🔹 accesso immediato da RAM
        x1 = signals[seg_idx][:16]
        x2 = signals[seg_idx + 1][:16]
        x_8s = np.concatenate([x1, x2], axis=-1)
        x_200 = resample_poly(x_8s, up=4, down=5, axis=-1)
        x_200 = x_200 / (np.quantile(np.abs(x_200), q=0.95, axis=-1, keepdims=True) + 1e-8)
        return {"x": torch.tensor(x_200, dtype=torch.float32),
                "y": torch.tensor(label, dtype=torch.long)}

# =====================================================
# Loader con opzione di bilanciamento
# =====================================================

def make_loader(patient_ids, dataset_path, gt_path, config, shuffle=True, balanced=False):
    dataset = CHBMITAllSegmentsLabeledDataset(
        patient_ids=patient_ids,
        data_dir=dataset_path,
        gt_dir=gt_path,
        segment_duration_sec=config.get("segment_duration_sec", 4),
        transform=None
    )

    if balanced:
        targets = [label for _, _, label, _ in dataset.index]
        class_counts = np.bincount(targets)
        class_weights = 1. / class_counts
        sample_weights = [class_weights[t] for t in targets]

        sampler = WeightedRandomSampler(
            weights=torch.DoubleTensor(sample_weights),
            num_samples=len(sample_weights),
            replacement=True
        )

        loader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            sampler=sampler,
                            num_workers=config["num_workers"],
                            pin_memory=True)
    else:
        loader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=shuffle,
                            num_workers=config["num_workers"],
                            pin_memory=False)

    return loader


# =====================================================
# MAIN di verifica distribuzione e forma dei dati
# =====================================================

if __name__ == "__main__":
    # Split dei pazienti
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients   = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients  = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]

    config = load_config("configs/finetuning.yml")
    dataset_path = config["dataset_path"]
    gt_path = "../../Datasets/chb_mit/GT"

    # Loader
    loader_train = make_loader(train_patients, dataset_path, gt_path, config,
                               shuffle=True, balanced=False)
    loader_val   = make_loader(val_patients, dataset_path, gt_path, config,
                               shuffle=False)
    loader_test  = make_loader(test_patients, dataset_path, gt_path, config,
                               shuffle=False)

    train_set = loader_train.dataset
    val_set   = loader_val.dataset
    test_set  = loader_test.dataset

    # =====================================================
    # Report dataset
    # =====================================================
    print("\n=== Dimensione dei set ===")
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    # Distribuzione label
    def count_labels(ds, name):
        labels = [label for _, _, label, _ in ds.index]
        num_pos = sum(1 for l in labels if l == 1)
        num_neg = sum(1 for l in labels if l == 0)
        ratio = num_pos / num_neg if num_neg > 0 else 0
        print(f"{name} --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {ratio:.6f}")

    print("\n=== Distribuzione etichette ===")
    count_labels(train_set, "TRAIN")
    count_labels(val_set, "VAL")
    count_labels(test_set, "TEST")

    # Controllo forma
    print("\n=== Controllo forma dei campioni ===")
    for name, ds in [("TRAIN", train_set), ("VAL", val_set), ("TEST", test_set)]:
        sample = ds[0]
        x0, y0 = sample["x"], sample["y"]
        print(f"{name} --- Sample shape: {tuple(x0.shape)} | Label: {y0.item()}")

    # Durata effettiva finestra
    x0 = train_set[0]["x"].numpy()
    durata = x0.shape[1] / 200
    print(f"\nDurata stimata finestra: {durata:.2f}s (atteso ≈ 8.0s)")