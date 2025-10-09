import os
import torch
import h5py
import numpy as np
import pandas as pd
from utils import load_config
from torch.utils.data import Dataset, DataLoader, DistributedSampler, WeightedRandomSampler

# seizure detection labeling senza oversampling e undersampling

# class CHBMITAllSegmentsLabeledDataset(Dataset):
#     def __init__(self, patient_ids, data_dir, gt_dir,
#                  segment_duration_sec=4, transform=None):
#         self.index = []  # (fpath, seg_idx, label, file_id)
#         self.transform = transform
#         self.segment_duration_sec = segment_duration_sec

#         for patient in patient_ids:
#             patient_folder = os.path.join(data_dir, patient)
#             gt_file = os.path.join(gt_dir, f"{patient}.csv")
#             if not os.path.exists(gt_file):
#                 print(f" GT not found for {patient}, skipping")
#                 continue

#             # parsing ground truth CSV
#             gt_df = pd.read_csv(gt_file, sep=';', engine='python')
#             seizure_map = {}
            
#             for _, row in gt_df.iterrows():
#                 edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
#                 if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
#                     seizure_map[edf_base] = []
#                 else:
#                     intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
#                     seizure_map[edf_base] = intervals

#             # scansiona gli h5 del paziente
#             for fname in sorted(os.listdir(patient_folder)):
#                 if not fname.endswith(".h5"):
#                     continue
#                 edf_base = fname.replace(".h5", "").replace("eeg_", "")
#                 fpath = os.path.join(patient_folder, fname)

#                 with h5py.File(fpath, 'r') as f:
#                     n_segments = f['signals'].shape[0]

#                 intervals = seizure_map.get(edf_base, [])

#                 # crea indice dei segmenti
#                 for i in range(n_segments):
#                     seg_start = i * self.segment_duration_sec
#                     seg_end = seg_start + self.segment_duration_sec
#                     label = 0
#                     for (st, en) in intervals:
#                         # LOGICA PAPER: positivo se inizio o fine dentro crisi
#                         if (seg_start >= st and seg_start < en) or (seg_end > st and seg_end <= en):
#                             label = 1
#                             break
#                     self.index.append((fpath, i, label, edf_base))

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
#         fpath, i, label, file_id = self.index[idx]
#         with h5py.File(fpath, 'r') as f:
#             x = f['signals'][i][:18]  # (channels, time)

#         # normalizzazione percentile 95 per canale
#         x = x / (np.quantile(np.abs(x), q=0.95, axis=-1, keepdims=True) + 1e-8)

#         x = torch.tensor(x, dtype=torch.float32)

#         if self.transform:
#             x = self.transform(x)
#         return {"x": x, "y": torch.tensor(label, dtype=torch.long), "file": file_id}

# def make_loader(patient_ids, dataset_path, gt_path, config, shuffle=True, balanced=False):

#     dataset = CHBMITAllSegmentsLabeledDataset(
#         patient_ids=patient_ids,
#         data_dir=dataset_path,
#         gt_dir=gt_path,
#         segment_duration_sec=config.get("segment_duration_sec", 4),
#         transform=None
#     )

#     if balanced:
#         targets = [label for _, _, label, _ in dataset.index]
#         class_counts = np.bincount(targets)
#         class_weights = 1. / class_counts
#         sample_weights = [class_weights[t] for t in targets]

#         sampler = WeightedRandomSampler(
#             weights=torch.DoubleTensor(sample_weights),
#             num_samples=len(sample_weights),
#             replacement=True
#         )

#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             sampler=sampler,
#                             num_workers=config["num_workers"],
#                             pin_memory=True)
   
#     else:
#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             shuffle=shuffle,
#                             num_workers=config["num_workers"],
#                             pin_memory=False)

#     return loader


# if __name__ == "__main__":
#     # Example usage
#     # Pazienti CHB01 - CHB19 per il training
#     train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
#     # CHB20 e CHB21 per validazione
#     val_patients = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
#     # CHB22 e CHB23 per test
#     test_patients = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
#     config = load_config("configs/finetuning.yml")

#     dataset_path = config["dataset_path"]
#     gt_path = "../../Datasets/chb_mit/GT"



#     loader_train = make_loader(train_patients, dataset_path, gt_path, config,
#                             shuffle=True, balanced=False)  
#     loader_val   = make_loader(val_patients, dataset_path, gt_path, config,
#                            shuffle=False)  
#     loader_test  = make_loader(test_patients, dataset_path, gt_path, config,
#                            shuffle=False) 
   
#     train_set = loader_train.dataset
#     val_set = loader_val.dataset
#     test_set = loader_test.dataset
    
#     print(f"Train set: {len(train_set)} samples")
#     print(f"Validation set: {len(val_set)} samples")
#     print(f"Test set: {len(test_set)} samples")


#     num_pos = sum([1 for *_, label, _ in train_set.index if label == 1])
#     num_neg = sum([1 for *_, label, _ in train_set.index if label == 0])
#     print(f"TRAIN --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

#     num_pos = sum([1 for *_, label, _ in val_set.index if label == 1])
#     num_neg = sum([1 for *_, label, _ in val_set.index if label == 0])
#     print(f"VAL --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

#     num_pos = sum([1 for *_, label, _ in test_set.index if label == 1])
#     num_neg = sum([1 for *_, label, _ in test_set.index if label == 0])
#     print(f"TEST --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")


# seizure detection labeling senza oversampling e undersampling

import os
import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
from scipy.signal import resample  # per il downsampling


# ------------------------------
# Data Augmentation per i positivi
# ------------------------------
class EEGAugment:
    def __init__(self, p_jitter=0.5, p_scale=0.5, p_mask=0.3,
                 jitter_std=0.01, scale_range=(0.9, 1.1), mask_max_ratio=0.15):
        self.p_jitter = p_jitter
        self.p_scale = p_scale
        self.p_mask = p_mask
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.mask_max_ratio = mask_max_ratio

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        C, T = x.shape[-2], x.shape[-1]

        if random.random() < self.p_jitter:
            x = x + self.jitter_std * torch.randn_like(x)

        if random.random() < self.p_scale:
            low, high = self.scale_range
            scales = torch.empty(C, 1, device=x.device).uniform_(low, high)
            x = x * scales

        if random.random() < self.p_mask:
            max_w = int(self.mask_max_ratio * T)
            if max_w > 0:
                w = random.randint(1, max_w)
                s = random.randint(0, max(0, T - w))
                x = x.clone()
                x[:, s:s+w] = 0.0
        return x


# ------------------------------
# Dataset CHB-MIT con runtime windows 10s@200Hz
# ------------------------------
class CHBMITAllSegmentsLabeledDataset(Dataset):
    def __init__(self, patient_ids, data_dir, gt_dir,
                 win_sec=10, step_sec=5, orig_fs=250, new_fs=200,
                 transform=None, pos_oversample_k=0, neg_undersample_ratio=None):

        self.index = []  # (x_window, label)
        self.transform = transform
        self.pos_oversample_k = int(pos_oversample_k)

        for patient in patient_ids:
            patient_folder = os.path.join(data_dir, patient)
            gt_file = os.path.join(gt_dir, f"{patient}.csv")
            if not os.path.exists(gt_file):
                print(f"[WARN] GT not found for {patient}, skipping")
                continue

            # parsing ground truth CSV
            gt_df = pd.read_csv(gt_file, sep=';', engine='python')
            seizure_map = {}
            for _, row in gt_df.iterrows():
                edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
                if isinstance(row.get("class_name", ""), str) and "no seizure" in row["class_name"].lower():
                    seizure_map[edf_base] = []
                else:
                    intervals = self.parse_intervals(row.get("Start (sec)"), row.get("End (sec)"))
                    seizure_map[edf_base] = intervals

            # scansiona gli h5 del paziente
            if not os.path.isdir(patient_folder):
                print(f"[WARN] Data folder not found for {patient}: {patient_folder}")
                continue

            for fname in sorted(os.listdir(patient_folder)):
                if not fname.endswith(".h5"):
                    continue
                edf_base = fname.replace(".h5", "").replace("eeg_", "")
                fpath = os.path.join(patient_folder, fname)
                intervals = seizure_map.get(edf_base, [])

                with h5py.File(fpath, 'r') as f:
                    segs = f['signals'][:]  # (n_segments, C, Tseg)
                    X250 = np.concatenate([seg[:18] for seg in segs], axis=-1)

                # --- resample UNA volta a 200 Hz ---
                Ttot_250 = X250.shape[-1]
                Ttot_200 = int(round(Ttot_250 * new_fs / orig_fs))
                X200 = resample(X250, Ttot_200, axis=-1)  # (C, T@200Hz)

                win_len = int(win_sec * new_fs)        
                step_std = win_len                        
                pos_step = int(5 * new_fs)                
                total_len = X200.shape[-1]

                added_starts = set()

                # 1) finestre NON sovrapposte su tutto il tracciato
                for start in range(0, total_len - win_len + 1, step_std):
                    end = start + win_len
                    x_win = X200[:, start:end]
                    # normalizzazione percentile 95
                    x_win = x_win / (np.quantile(np.abs(x_win), q=0.95, axis=-1, keepdims=True) + 1e-8)

                    
                    start_sec, end_sec = start / new_fs, end / new_fs
                    label = 0
                    for (st, en) in intervals:
                        if not (end_sec <= st or start_sec >= en):
                            label = 1
                            break

                    self.index.append((x_win.astype(np.float32), label))
                    added_starts.add(start)

                # 2) finestre AGGIUNTIVE con step=5s SOLO dentro (o a cavallo di) crisi
                for (st, en) in intervals:
                    
                    
                    start_min = max(0, int((st * new_fs) - (win_len - pos_step)))
                    start_max = min(total_len - win_len, int(en * new_fs))
                    # allineiamo a griglia 5s
                    start_5s = start_min - (start_min % pos_step)
                    for start in range(start_5s, start_max + 1, pos_step):
                        if start in added_starts:
                            continue  # finestra già aggiunta dallo stream non-overlap
                        end = start + win_len
                        if end > total_len:
                            continue
                        start_sec, end_sec = start / new_fs, end / new_fs
                        # aggiungi SOLO se davvero overlappa la crisi
                        overlaps = any(not (end_sec <= st_i or start_sec >= en_i) for (st_i, en_i) in intervals)
                        if not overlaps:
                            continue

                        x_win = X200[:, start:end]
                        x_win = x_win / (np.quantile(np.abs(x_win), q=0.95, axis=-1, keepdims=True) + 1e-8)
                        self.index.append((x_win.astype(np.float32), 1))  # forziamo label=1 per le aggiunte positive
                        added_starts.add(start)
                        if self.pos_oversample_k > 0:
                            for _ in range(self.pos_oversample_k):
                                self.index.append((x_win.astype(np.float32), 1))

        # undersampling negativi
        if neg_undersample_ratio is not None and neg_undersample_ratio < 1.0:
            pos_samples = [x for x in self.index if x[1] == 1]
            neg_samples = [x for x in self.index if x[1] == 0]
            keep_n = int(len(neg_samples) * neg_undersample_ratio)
            neg_samples = random.sample(neg_samples, keep_n)
            self.index = pos_samples + neg_samples
            random.shuffle(self.index)

    def parse_intervals(self, start_val, end_val):
        intervals = []
        if pd.isna(start_val) or pd.isna(end_val):
            return intervals
        start_parts = str(start_val).split(',')
        end_parts = str(end_val).split(',')
        for s_group, e_group in zip(start_parts, end_parts):
            starts = [float(x) for x in s_group.split('-') if x.strip()]
            ends = [float(x) for x in e_group.split('-') if x.strip()]
            for st, en in zip(starts, ends):
                intervals.append((st, en))
        return intervals

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        x, label = self.index[idx]
        x = torch.tensor(x, dtype=torch.float32)
        if self.transform is not None and label == 1:
            x = self.transform(x)
        return {"x": x, "y": torch.tensor(label, dtype=torch.long)}


# ------------------------------
# make_loader
# ------------------------------
def make_loader(patient_ids, dataset_path, gt_path, config,
                shuffle=True, balanced=False, pos_oversample_k=0,
                transform=None, neg_undersample_ratio=None):

    dataset = CHBMITAllSegmentsLabeledDataset(
        patient_ids=patient_ids,
        data_dir=dataset_path,
        gt_dir=gt_path,
        win_sec=10,
        step_sec=10,
        orig_fs=250,
        new_fs=200,
        transform=transform,
        pos_oversample_k=pos_oversample_k,
        neg_undersample_ratio=neg_undersample_ratio
    )

    if balanced:
        targets = [label for _, label in dataset.index]
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
                            pin_memory=False)
    else:
        loader = DataLoader(dataset,
                            batch_size=config["batch_size"],
                            shuffle=shuffle,
                            num_workers=config["num_workers"],
                            pin_memory=False)
    return loader


# ------------------------------
# Debug main
# ------------------------------
if __name__ == "__main__":
    from utils import load_config
    config = load_config("configs/finetuning.yml")

    dataset_path = config["dataset_path"]
    gt_path = "../../Datasets/chb_mit/GT"

    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients   = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients  = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]

    augment_pos = EEGAugment(p_jitter=0.7, p_scale=0.5, p_mask=0.3, jitter_std=0.02)

    loader_train = make_loader(train_patients, dataset_path, gt_path,config,
                               shuffle=True, balanced=False,
                               pos_oversample_k=4, transform=augment_pos, neg_undersample_ratio=0.27) 

    loader_val   = make_loader(val_patients, dataset_path, gt_path, config,
                               shuffle=False, balanced=False,
                               pos_oversample_k=0, transform=None)

    loader_test  = make_loader(test_patients, dataset_path, gt_path, config,
                               shuffle=False, balanced=False,
                               pos_oversample_k=0, transform=None)

    # Debug numeri e shape
    for name, ds in [("TRAIN", loader_train.dataset),
                     ("VAL", loader_val.dataset),
                     ("TEST", loader_test.dataset)]:
        labels = [lbl for _, lbl in ds.index]
        num_pos = sum(1 for lbl in labels if lbl == 1)
        num_neg = sum(1 for lbl in labels if lbl == 0)
        ratio = num_pos / (num_neg + 1e-8)
        # prendi un campione qualsiasi
        x0, y0 = ds[0]["x"], ds[0]["y"]
        print(f"{name} --- Total: {len(ds)} | Positives: {num_pos}, Negatives: {num_neg}, "
              f"Ratio: {ratio:.6f} | Sample shape: {tuple(x0.shape)} | Label: {y0.item()}")

# ------------------------------ ------------------------------ ------------------------------
# preictal vs ictal labeling

# import os
# import torch
# import h5py
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader, DistributedSampler
# from utils import load_config
# from torch.utils.data import WeightedRandomSampler


# class CHBMITAllSegmentsLabeledDataset(Dataset):
#     def __init__(self, patient_ids, data_dir, gt_dir,
#                  segment_duration_sec=4, mean=None, std=None, transform=None,
#                  preictal_seconds=300, min_overlap_ratio=0.5):
#         self.index = []  # (fpath, seg_idx, label, file_id)
#         self.transform = transform
#         self.segment_duration_sec = segment_duration_sec
#         self.preictal_seconds = preictal_seconds
#         self.min_overlap_ratio = min_overlap_ratio

#         self.mean = torch.as_tensor(mean, dtype=torch.float32) if mean is not None else None
#         self.std  = torch.as_tensor(std,  dtype=torch.float32)  if std  is not None else None
#         if self.mean is not None and self.mean.ndim == 1:
#             self.mean = self.mean[:, None]
#         if self.std is not None and self.std.ndim == 1:
#             self.std = self.std[:, None]
#         self.eps = 1e-6

#         for patient in patient_ids:
#             patient_folder = os.path.join(data_dir, patient)
#             gt_file = os.path.join(gt_dir, f"{patient}.csv")
#             if not os.path.exists(gt_file):
#                 print(f" GT not found for {patient}, skipping")
#                 continue

#             # parsing ground truth CSV
#             gt_df = pd.read_csv(gt_file, sep=';', engine='python')
#             seizure_map = {}

#             for _, row in gt_df.iterrows():
#                 edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
#                 if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
#                     seizure_map[edf_base] = []
#                 else:
#                     intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
#                     seizure_map[edf_base] = intervals

#             # Scorri gli h5 nella cartella del paziente
#             for fname in sorted(os.listdir(patient_folder)):
#                 if not fname.endswith(".h5"):
#                     continue
#                 edf_base = fname.replace(".h5", "").replace("eeg_", "")
#                 fpath = os.path.join(patient_folder, fname)

#                 with h5py.File(fpath, 'r') as f:
#                     n_segments = f['signals'].shape[0]

#                 intervals = seizure_map.get(edf_base, [])

#                 for i in range(n_segments):
#                     seg_start = i * self.segment_duration_sec
#                     seg_end   = seg_start + self.segment_duration_sec

#                     label = self.assign_label(seg_start, seg_end, intervals)

#                     if label is not None:  # tiene solo preictal o ictal
#                         self.index.append((fpath, i, label, edf_base))

#     def assign_label(self, seg_start, seg_end, intervals):
#         """Ritorna 1=ictal, 0=preictal, None=interictal"""
#         window_len = seg_end - seg_start

#         # ictal
#         for (st, en) in intervals:
#             overlap_start = max(seg_start, st)
#             overlap_end   = min(seg_end, en)
#             overlap = max(0, overlap_end - overlap_start)
#             if overlap / window_len >= self.min_overlap_ratio:
#                 return 1

#         # preictal
#         for (st, en) in intervals:
#             pre_start = max(0, st - self.preictal_seconds)
#             overlap_start = max(seg_start, pre_start)
#             overlap_end   = min(seg_end, st)
#             overlap = max(0, overlap_end - overlap_start)
#             if overlap / window_len >= self.min_overlap_ratio:
#                 return 0

#         return None  # interictal scartato

#     def parse_intervals(self, start_val, end_val):
#         intervals = []
#         if pd.isna(start_val) or pd.isna(end_val):
#             return intervals
#         start_parts = str(start_val).split(',')
#         end_parts   = str(end_val).split(',')
#         for s_group, e_group in zip(start_parts, end_parts):
#             starts = [float(x) for x in s_group.split('-') if x.strip() not in ["", "0"]]
#             ends   = [float(x) for x in e_group.split('-') if x.strip() not in ["", "0"]]
#             for st, en in zip(starts, ends):
#                 intervals.append((st, en))
#         return intervals

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         fpath, i, label, file_id = self.index[idx]
#         with h5py.File(fpath, 'r') as f:
#             x = f['signals'][i][:18]  # (channels, time)
#         x = torch.tensor(x, dtype=torch.float32)

#         if self.mean is not None and self.std is not None:
#             x = (x - self.mean) / (self.std + self.eps)

#         if self.transform:
#             x = self.transform(x)
#         return {"x": x, "y": torch.tensor(label, dtype=torch.long), "file": file_id}


# def make_loader(patient_ids, dataset_path, gt_path, config, mean=None, std=None,
#                 shuffle=True, is_ddp=False, rank=0, world_size=1, balanced=False):

#     dataset = CHBMITAllSegmentsLabeledDataset(
#         patient_ids=patient_ids,
#         data_dir=dataset_path,
#         gt_dir=gt_path,
#         segment_duration_sec=config.get("segment_duration_sec", 4),
#         mean=mean,
#         std=std,
#         transform=None
#     )

#     if balanced:
#         targets = [label for _, _, label, _ in dataset.index]
#         class_counts = np.bincount(targets)
#         class_weights = 1. / class_counts
#         sample_weights = [class_weights[t] for t in targets]

#         sampler = WeightedRandomSampler(
#             weights=torch.DoubleTensor(sample_weights),
#             num_samples=len(sample_weights),
#             replacement=True
#         )

#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             sampler=sampler,
#                             num_workers=config["num_workers"],
#                             pin_memory=True)
#     elif is_ddp:
#         sampler = DistributedSampler(dataset,
#                                     num_replicas=world_size,
#                                     rank=rank,
#                                     shuffle=shuffle)
#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             sampler=sampler,
#                             num_workers=config["num_workers"],
#                             pin_memory=True)
#     else:
#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             shuffle=shuffle,
#                             num_workers=config["num_workers"],
#                             pin_memory=True)

#     return loader


# if __name__ == "__main__":
#     train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
#     val_patients = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
#     test_patients = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
#     config = load_config("configs/finetuning.yml")

#     dataset_path = config["dataset_path"]
#     gt_path = "../../Datasets/chb_mit/GT"

#     loader_train = make_loader(train_patients, dataset_path, gt_path, config, shuffle=True)
#     loader_val = make_loader(val_patients, dataset_path, gt_path, config, shuffle=False)
#     loader_test = make_loader(test_patients, dataset_path, gt_path, config, shuffle=False)

#     train_set = loader_train.dataset
#     val_set = loader_val.dataset
#     test_set = loader_test.dataset
    
#     print(f"Train set: {len(train_set)} samples")
#     print(f"Validation set: {len(val_set)} samples")
#     print(f"Test set: {len(test_set)} samples")

#     num_pos = sum([1 for _,_,label,_ in train_set.index if label==1])
#     num_neg = sum([1 for _,_,label,_ in train_set.index if label==0])
#     print(f"TRAIN --- Ictal: {num_pos}, Preictal: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

#     num_pos = sum([1 for _,_,label,_ in val_set.index if label==1])
#     num_neg = sum([1 for _,_,label,_ in val_set.index if label==0])
#     print(f"VAL --- Ictal: {num_pos}, Preictal: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

#     num_pos = sum([1 for _,_,label,_ in test_set.index if label==1])
#     num_neg = sum([1 for _,_,label,_ in test_set.index if label==0])
#     print(f"TEST --- Ictal: {num_pos}, Preictal: {num_neg}, Ratio: {num_pos/num_neg:.6f}")



# import os
# import torch
# import h5py
# import numpy as np
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from scipy.signal import resample
# import random
# from utils import load_config


# # ------------------------------
# # Dataset CHB-MIT con runtime windows 10s@200Hz
# # ------------------------------
# class CHBMITRuntimeDataset(Dataset):
#     def __init__(self, patient_ids, data_dir, gt_dir,
#                  win_sec=10, step_sec=10, orig_fs=250, new_fs=200,
#                  transform=None):
#         self.index = []  # (np.array, label, file_id)
#         self.transform = transform

#         for patient in patient_ids:
#             patient_folder = os.path.join(data_dir, patient)
#             gt_file = os.path.join(gt_dir, f"{patient}.csv")
#             if not os.path.exists(gt_file):
#                 print(f"[WARN] GT not found for {patient}, skipping")
#                 continue

#             # parsing ground truth CSV
#             gt_df = pd.read_csv(gt_file, sep=';', engine='python')
#             seizure_map = {}
#             for _, row in gt_df.iterrows():
#                 edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
#                 if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
#                     seizure_map[edf_base] = []
#                 else:
#                     intervals = self.parse_intervals(row["Start (sec)"], row["End (sec)"])
#                     seizure_map[edf_base] = intervals

#             # scansiona h5
#             if not os.path.isdir(patient_folder):
#                 print(f"[WARN] Data folder not found for {patient}: {patient_folder}")
#                 continue

#             for fname in sorted(os.listdir(patient_folder)):
#                 if not fname.endswith(".h5"):
#                     continue
#                 edf_base = fname.replace(".h5", "").replace("eeg_", "")
#                 fpath = os.path.join(patient_folder, fname)
#                 intervals = seizure_map.get(edf_base, [])

#                 with h5py.File(fpath, 'r') as f:
#                     segs = f['signals'][:]  # (n_segments, C, Tseg)
#                     # concatena i segmenti (C, Ttot)
#                     X = np.concatenate([seg[:18] for seg in segs], axis=-1)

#                 win_len = win_sec * orig_fs
#                 step_len = step_sec * orig_fs
#                 total_len = X.shape[-1]

#                 for start in range(0, total_len - win_len + 1, step_len):
#                     end = start + win_len
#                     x_win = X[:, start:end]
#                     # downsample 250→200 Hz
#                     x_ds = resample(x_win, win_sec * new_fs, axis=-1)
#                     # normalizzazione percentile 95 per canale
#                     x_ds = x_ds / (np.quantile(np.abs(x_ds), q=0.95, axis=-1, keepdims=True) + 1e-8)
#                     # label = positivo se la finestra interseca una crisi
#                     label = 0
#                     for (st, en) in intervals:
#                         if not (end/orig_fs <= st or start/orig_fs >= en):
#                             label = 1
#                             break
#                     self.index.append((x_ds.astype(np.float32), label, edf_base))

#     def parse_intervals(self, start_val, end_val):
#         intervals = []
#         if pd.isna(start_val) or pd.isna(end_val):
#             return intervals
#         start_parts = str(start_val).split(',')
#         end_parts = str(end_val).split(',')
#         for s_group, e_group in zip(start_parts, end_parts):
#             starts = [float(x) for x in s_group.split('-') if x.strip()]
#             ends = [float(x) for x in e_group.split('-') if x.strip()]
#             for st, en in zip(starts, ends):
#                 intervals.append((st, en))
#         return intervals

#     def __len__(self): 
#         return len(self.index)

#     def __getitem__(self, idx):
#         x, label, file_id = self.index[idx]
#         x = torch.tensor(x, dtype=torch.float32)
#         if self.transform:
#             x = self.transform(x)
#         return {"x": x, "y": torch.tensor(label, dtype=torch.long), "file": file_id}


# # ------------------------------
# # make_loader
# # ------------------------------
# def make_loader(patient_ids, dataset_path, gt_path, config,
#                 shuffle=True, balanced=False):

#     dataset = CHBMITRuntimeDataset(
#         patient_ids=patient_ids,
#         data_dir=dataset_path,
#         gt_dir=gt_path,
#         win_sec=config.get("win_sec", 10),
#         step_sec=config.get("step_sec", 10),
#         orig_fs=250,
#         new_fs=200,
#         transform=None
#     )

#     if balanced:
#         targets = [label for _, label, _ in dataset.index]
#         class_counts = np.bincount(targets)
#         class_weights = 1. / class_counts
#         sample_weights = [class_weights[t] for t in targets]
#         sampler = WeightedRandomSampler(
#             weights=torch.DoubleTensor(sample_weights),
#             num_samples=len(sample_weights),
#             replacement=True
#         )
#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             sampler=sampler,
#                             num_workers=config["num_workers"],
#                             pin_memory=True)
#     else:
#         loader = DataLoader(dataset,
#                             batch_size=config["batch_size"],
#                             shuffle=shuffle,
#                             num_workers=config["num_workers"],
#                             pin_memory=False)
#     return loader


# # ------------------------------
# # Debug main
# # ------------------------------
# if __name__ == "__main__":
#     train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
#     val_patients   = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
#     test_patients  = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
#     config = load_config("configs/finetuning.yml")

#     dataset_path = config["dataset_path"]
#     gt_path = "../../Datasets/chb_mit/GT"

#     loader_train = make_loader(train_patients, dataset_path, gt_path, config, shuffle=True, balanced=False)
#     loader_val   = make_loader(val_patients, dataset_path, gt_path, config, shuffle=False)
#     loader_test  = make_loader(test_patients, dataset_path, gt_path, config, shuffle=False)

#     for name, ds in [("TRAIN", loader_train.dataset),
#                      ("VAL", loader_val.dataset),
#                      ("TEST", loader_test.dataset)]:
#         labels = [lbl for _, lbl, _ in ds.index]
#         num_pos = sum(1 for l in labels if l == 1)
#         num_neg = sum(1 for l in labels if l == 0)
#         ratio = num_pos/(num_neg+1e-8)
#         x0 = ds[0]["x"]
#         print(f"{name} --- Total: {len(ds)} | Positives: {num_pos}, Negatives: {num_neg}, "
#               f"Ratio: {ratio:.6f} | Sample shape: {tuple(x0.shape)}")