
import os
import torch
import h5py
import numpy as np
import pandas as pd
from utils import load_config
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler
from scipy.signal import resample_poly  # per il downsampling

# seizure detection labeling senza oversampling e undersampling

class CHBMITAllSegmentsLabeledDataset(Dataset):
    def __init__(self, patient_ids, data_dir, gt_dir,
                 segment_duration_sec=4, transform=None):
        self.index = []  # (fpath, seg_idx, label, file_id)
        self.transform = transform
        self.segment_duration_sec = segment_duration_sec

        for patient in patient_ids:
            patient_folder = os.path.join(data_dir, patient)
            gt_file = os.path.join(gt_dir, f"{patient}.csv")
            if not os.path.exists(gt_file):
                print(f" GT not found for {patient}, skipping")
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

            # scansiona gli h5 del paziente
            for fname in sorted(os.listdir(patient_folder)):
                if not fname.endswith(".h5"):
                    continue
                edf_base = fname.replace(".h5", "").replace("eeg_", "")
                fpath = os.path.join(patient_folder, fname)

                with h5py.File(fpath, 'r') as f:
                    n_segments = f['signals'].shape[0]

                intervals = seizure_map.get(edf_base, [])

                # crea indice dei segmenti
                for i in range(n_segments):
                    seg_start = i * self.segment_duration_sec
                    seg_end = seg_start + self.segment_duration_sec
                    label = 0
                    for (st, en) in intervals:
                        
                        if (seg_start >= st and seg_start < en) or (seg_end > st and seg_end <= en):
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

    # def __getitem__(self, idx):
    #     fpath, i, label, file_id = self.index[idx]
    #     with h5py.File(fpath, 'r') as f:
    #         x = f['signals'][i][:16]  # (channels, time)
            

    #         target_fs = 200
    #         orig_fs = 250
    #         x_200 = resample_poly(x, up=target_fs, down=orig_fs, axis=-1)

             

    #     # normalizzazione percentile 95 per canale
    #     x_200 = x_200 / (np.quantile(np.abs(x_200), q=0.95, axis=-1, keepdims=True) + 1e-8)

    #     x_200 = torch.tensor(x_200, dtype=torch.float32)

    #     if self.transform:
    #         x_200 = self.transform(x_200)

    #     return {"x": x_200, "y": torch.tensor(label, dtype=torch.long)}

    def __getitem__(self, idx):
        fpath, i, label, file_id = self.index[idx]

        with h5py.File(fpath, 'r') as f:
            x1 = f['signals'][i][:16]  # primo segmento 4s a 250Hz

            # tenta di prendere il segmento successivo nello stesso file
            n_segments = f['signals'].shape[0]
            if i + 1 < n_segments:
                x2 = f['signals'][i + 1][:16]
            else:
                # se non c’è un segmento successivo, duplica l’ultimo
                x2 = x1

            # concatena → 8 secondi a 250Hz
            x_8s = np.concatenate([x1, x2], axis=-1)

            # downsampling a 200Hz
            target_fs = 200
            orig_fs = 250
            x_200 = resample_poly(x_8s, up=target_fs, down=orig_fs, axis=-1)

        # normalizzazione percentile 95 per canale
        x_200 = x_200 / (np.quantile(np.abs(x_200), q=0.95, axis=-1, keepdims=True) + 1e-8)
        x_200 = torch.tensor(x_200, dtype=torch.float32)

        if self.transform:
            x_200 = self.transform(x_200)

        # etichetta: se uno dei due segmenti contiene crisi → label = 1
        label_next = 0
        if i + 1 < len(self.index) and self.index[i + 1][3] == file_id:
            label_next = self.index[i + 1][2]
        label_8s = max(label, label_next)

        return {"x": x_200, "y": torch.tensor(label_8s, dtype=torch.long)}

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


if __name__ == "__main__":
    # Example usage
    # Pazienti CHB01 - CHB19 per il training
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    # CHB20 e CHB21 per validazione
    val_patients = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    # CHB22 e CHB23 per test
    test_patients = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    config = load_config("configs/finetuning.yml")

    dataset_path = config["dataset_path"]
    gt_path = "../../Datasets/chb_mit/GT"



    loader_train = make_loader(train_patients, dataset_path, gt_path, config,
                            shuffle=True, balanced=False)  
    loader_val   = make_loader(val_patients, dataset_path, gt_path, config,
                           shuffle=False)  
    loader_test  = make_loader(test_patients, dataset_path, gt_path, config,
                           shuffle=False) 
   
    train_set = loader_train.dataset
    val_set = loader_val.dataset
    test_set = loader_test.dataset
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")


    num_pos = sum([1 for *_, label, _ in train_set.index if label == 1])
    num_neg = sum([1 for *_, label, _ in train_set.index if label == 0])
    print(f"TRAIN --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

    num_pos = sum([1 for *_, label, _ in val_set.index if label == 1])
    num_neg = sum([1 for *_, label, _ in val_set.index if label == 0])
    print(f"VAL --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

    num_pos = sum([1 for *_, label, _ in test_set.index if label == 1])
    num_neg = sum([1 for *_, label, _ in test_set.index if label == 0])
    print(f"TEST --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

    # prendi un campione qualsiasi
    for  name, ds in [("TRAIN", train_set), ("VAL", val_set), ("TEST", test_set)]:
        
        x0, y0 = ds[0]["x"], ds[0]["y"]
        print(f"{name} --- Sample shape: {tuple(x0.shape)} | Label: {y0.item()}")