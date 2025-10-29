import os
import torch
import h5py
import numpy as np
import pandas as pd
from utils import load_config
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


# =====================================================
# Dataset CHB-MIT con segmenti da 4s a 250 Hz
# =====================================================

class CHBMITAllSegmentsLabeledDataset(Dataset):
    def __init__(self, patient_ids, data_dir, gt_dir,
                 segment_duration_sec=8, transform=None):
        """
        Dataset CHB-MIT:
        - segmenti da 4s a 250 Hz (senza concatenazione)
        - labeling secondo gli intervalli di crisi nel CSV GT
        """
        self.index = []  # (fpath, seg_idx, label, file_id)
        self.transform = transform
        self.segment_duration_sec = segment_duration_sec

        for patient in patient_ids:
            patient_folder = os.path.join(data_dir, patient)
            gt_file = os.path.join(gt_dir, f"{patient}.csv")

            if not os.path.exists(gt_file):
                print(f"GT not found for {patient}, skipping")
                continue

            # parsing ground truth CSV
            gt_df = pd.read_csv(gt_file, sep=';', engine='python')
            seizure_map = {}

            for _, row in gt_df.iterrows():
                edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
                if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
                    seizure_map[edf_base] = []
                else:
                    intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
                    seizure_map[edf_base] = intervals

            # scansiona i file HDF5 del paziente
            for fname in sorted(os.listdir(patient_folder)):
                if not fname.endswith(".h5"):
                    continue

                edf_base = fname.replace(".h5", "").replace("eeg_", "")
                fpath = os.path.join(patient_folder, fname)

                with h5py.File(fpath, 'r') as f:
                    n_segments = f['signals'].shape[0]

                intervals = seizure_map.get(edf_base, [])

                # crea indice dei segmenti da 4s
                for i in range(n_segments):
                    seg_start = i * self.segment_duration_sec
                    seg_end = seg_start + self.segment_duration_sec
                    label = 0
                    for (st, en) in intervals:
                        # se l’intervallo del segmento cade in una crisi → label 1
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
        """
        Ritorna un segmento da 4s a 250Hz (16 canali, 1000 campioni)
        """
        fpath, seg_idx, label, file_id = self.index[idx]

        with h5py.File(fpath, 'r') as f:
            x = f['signals'][seg_idx][:18]  # (channels, time)

        # Normalizzazione percentile 95 per canale
        x = x / (np.quantile(np.abs(x), q=0.95, axis=-1, keepdims=True) + 1e-8)

        # Tensor conversion
        x = torch.tensor(x, dtype=torch.float32)

        if self.transform:
            x = self.transform(x)

        return {"x": x, "y": torch.tensor(label, dtype=torch.long)}


# =====================================================
# Loader (facoltativamente bilanciato)
# =====================================================

def make_loader(patient_ids, dataset_path, gt_path, config, shuffle=True, balanced=False, neg_to_pos_ratio=5):
    dataset = CHBMITAllSegmentsLabeledDataset(
        patient_ids=patient_ids,
        data_dir=dataset_path,
        gt_dir=gt_path,
        segment_duration_sec=config.get("segment_duration_sec", 8),
        transform=None
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

        #print(f"⚖️ Kept {num_pos} positives and {num_neg_to_keep} negatives "
        #      f"→ ratio {num_pos / num_neg_to_keep:.4f} ({num_pos+num_neg_to_keep} total)")
    
    # ============================
    #  DataLoader standard
    # ============================
    loader = DataLoader(dataset,
                        batch_size=config["batch_size"],
                        shuffle=shuffle,
                        num_workers=config["num_workers"],
                        pin_memory=False)
    return loader

# =====================================================
# MAIN – verifica dataset
# =====================================================

if __name__ == "__main__":
    # Split pazienti
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients   = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    test_patients  = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]

    config = load_config("configs/finetuning.yml")
    dataset_path = config["dataset_path_8s"]
    gt_path = "../../Datasets/chb_mit/GT"

    # Loader
    loader_train = make_loader(train_patients, dataset_path, gt_path, config,
                               shuffle=True, balanced=True, neg_to_pos_ratio=5)
    loader_val   = make_loader(val_patients, dataset_path, gt_path, config,
                               shuffle=False, balanced=False)
    loader_test  = make_loader(test_patients, dataset_path, gt_path, config,
                               shuffle=False)

    train_set = loader_train.dataset
    val_set   = loader_val.dataset
    test_set  = loader_test.dataset

    print(f"\nTrain set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    # Distribuzione etichette
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

    # Controllo forma dei campioni
    print("\n=== Controllo forma ===")
    for name, ds in [("TRAIN", train_set), ("VAL", val_set), ("TEST", test_set)]:
        sample = ds[0]
        x0, y0 = sample["x"], sample["y"]
        print(f"{name} --- shape: {tuple(x0.shape)} | label: {y0.item()}")

    # Verifica durata
    x0 = train_set[0]["x"].numpy()
    sec = x0.shape[1] / 250
    print(f"\nDurata stimata finestra: {sec:.2f}s (atteso ≈8.0s)")