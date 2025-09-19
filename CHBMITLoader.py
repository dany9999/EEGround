

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
#                  segment_duration_sec=4,mean = None, std=None, transform=None):
#         self.index = []  # (fpath, seg_idx, label, file_id)
#         self.transform = transform
#         self.segment_duration_sec = segment_duration_sec

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
#                 edf_base = fname.replace(".h5", "")
#                 edf_base = edf_base.replace("eeg_", "")
#                 fpath = os.path.join(patient_folder, fname)

#                 with h5py.File(fpath, 'r') as f:
#                     n_segments = f['signals'].shape[0]
               

#                 intervals = seizure_map.get(edf_base, [])

#                 # crea un indice di segmenti con etichetta
#                 min_overlap_ratio = 0.5  # es: almeno il 50% della finestra deve essere dentro la seizure

#                 for i in range(n_segments):
#                     seg_start = i * self.segment_duration_sec
#                     seg_end   = seg_start + self.segment_duration_sec

#                     label = 0
#                     for (st, en) in intervals:
#                         # Calcola overlap con finestra
#                         overlap_start = max(seg_start, st)
#                         overlap_end   = min(seg_end, en)
#                         overlap = max(0, overlap_end - overlap_start)

#                         window_len = seg_end - seg_start
#                         overlap_ratio = overlap / window_len

#                         if overlap_ratio >= min_overlap_ratio:
#                             label = 1
#                             break

#                     self.index.append((fpath, i, label, edf_base))

            

#     def parse_intervals(self, start_val, end_val):
#         """
#         Converte i campi Start/End in una lista di tuple [(st1, en1), (st2, en2), ...]
#         Supporta:
#         - singoli numeri: 3367
#         - intervalli multipli separati da trattini: 834-2378-3362
#         - più intervalli separati da virgola: 263-843-1524,318-1020-1595
#         """
#         intervals = []
#         if pd.isna(start_val) or pd.isna(end_val):
#             return intervals
#         start_parts = str(start_val).split(',')
#         end_parts   = str(end_val).split(',')
#         for s_group, e_group in zip(start_parts, end_parts):
#             starts = [float(x) for x in s_group.split('-') if x.strip() not in ["", "0"]]
#             ends   = [float(x) for x in e_group.split('-') if x.strip() not in ["", "0"]]
#             # Associa ciascun start con il corrispondente end
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

#         # normalizzazione globale
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
#             # Estrai le label dal dataset
#             targets = [label for _, _, label, _ in dataset.index]

#             class_counts = np.bincount(targets)
#             class_weights = 1. / class_counts
#             sample_weights = [class_weights[t] for t in targets]

#             sampler = WeightedRandomSampler(
#                 weights=torch.DoubleTensor(sample_weights),
#                 num_samples=len(sample_weights),
#                 replacement=True
#             )

#             loader = DataLoader(dataset,
#                                 batch_size=config["batch_size"],
#                                 sampler=sampler,
#                                 num_workers=config["num_workers"],
#                                 pin_memory=True)
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
#     print(f"TRAIN --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

#     num_pos = sum([1 for _,_,label,_ in val_set.index if label==1])
#     num_neg = sum([1 for _,_,label,_ in val_set.index if label==0])
#     print(f"VAL --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

#     num_pos = sum([1 for _,_,label,_ in test_set.index if label==1])
#     num_neg = sum([1 for _,_,label,_ in test_set.index if label==0])
#     print(f"TEST --- Positives: {num_pos}, Negatives: {num_neg}, Ratio: {num_pos/num_neg:.6f}")


import os
import torch
import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import load_config
from torch.utils.data import WeightedRandomSampler


class CHBMITAllSegmentsLabeledDataset(Dataset):
    def __init__(self, patient_ids, data_dir, gt_dir,
                 segment_duration_sec=4, mean=None, std=None, transform=None,
                 preictal_seconds=300, min_overlap_ratio=0.5):
        self.index = []  # (fpath, seg_idx, label, file_id)
        self.transform = transform
        self.segment_duration_sec = segment_duration_sec
        self.preictal_seconds = preictal_seconds
        self.min_overlap_ratio = min_overlap_ratio

        self.mean = torch.as_tensor(mean, dtype=torch.float32) if mean is not None else None
        self.std  = torch.as_tensor(std,  dtype=torch.float32)  if std  is not None else None
        if self.mean is not None and self.mean.ndim == 1:
            self.mean = self.mean[:, None]
        if self.std is not None and self.std.ndim == 1:
            self.std = self.std[:, None]
        self.eps = 1e-6

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

            # Scorri gli h5 nella cartella del paziente
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
                    seg_end   = seg_start + self.segment_duration_sec

                    label = self.assign_label(seg_start, seg_end, intervals)

                    if label is not None:  # tiene solo preictal o ictal
                        self.index.append((fpath, i, label, edf_base))

    def assign_label(self, seg_start, seg_end, intervals):
        """Ritorna 1=ictal, 0=preictal, None=interictal"""
        window_len = seg_end - seg_start

        # ictal
        for (st, en) in intervals:
            overlap_start = max(seg_start, st)
            overlap_end   = min(seg_end, en)
            overlap = max(0, overlap_end - overlap_start)
            if overlap / window_len >= self.min_overlap_ratio:
                return 1

        # preictal
        for (st, en) in intervals:
            pre_start = max(0, st - self.preictal_seconds)
            overlap_start = max(seg_start, pre_start)
            overlap_end   = min(seg_end, st)
            overlap = max(0, overlap_end - overlap_start)
            if overlap / window_len >= self.min_overlap_ratio:
                return 0

        return None  # interictal scartato

    def parse_intervals(self, start_val, end_val):
        intervals = []
        if pd.isna(start_val) or pd.isna(end_val):
            return intervals
        start_parts = str(start_val).split(',')
        end_parts   = str(end_val).split(',')
        for s_group, e_group in zip(start_parts, end_parts):
            starts = [float(x) for x in s_group.split('-') if x.strip() not in ["", "0"]]
            ends   = [float(x) for x in e_group.split('-') if x.strip() not in ["", "0"]]
            for st, en in zip(starts, ends):
                intervals.append((st, en))
        return intervals

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, i, label, file_id = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['signals'][i][:18]  # (channels, time)
        x = torch.tensor(x, dtype=torch.float32)

        if self.mean is not None and self.std is not None:
            x = (x - self.mean) / (self.std + self.eps)

        if self.transform:
            x = self.transform(x)
        return {"x": x, "y": torch.tensor(label, dtype=torch.long), "file": file_id}


def make_loader(patient_ids, dataset_path, gt_path, config, mean=None, std=None,
                shuffle=True, is_ddp=False, rank=0, world_size=1, balanced=False):

    dataset = CHBMITAllSegmentsLabeledDataset(
        patient_ids=patient_ids,
        data_dir=dataset_path,
        gt_dir=gt_path,
        segment_duration_sec=config.get("segment_duration_sec", 4),
        mean=mean,
        std=std,
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
    elif is_ddp:
        sampler = DistributedSampler(dataset,
                                    num_replicas=world_size,
                                    rank=rank,
                                    shuffle=shuffle)
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
                            pin_memory=True)

    return loader


if __name__ == "__main__":
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    config = load_config("configs/finetuning.yml")

    dataset_path = config["dataset_path"]
    gt_path = "../../Datasets/chb_mit/GT"

    loader_train = make_loader(train_patients, dataset_path, gt_path, config, shuffle=True)
    loader_val = make_loader(val_patients, dataset_path, gt_path, config, shuffle=False)
    loader_test = make_loader(test_patients, dataset_path, gt_path, config, shuffle=False)

    train_set = loader_train.dataset
    val_set = loader_val.dataset
    test_set = loader_test.dataset
    
    print(f"Train set: {len(train_set)} samples")
    print(f"Validation set: {len(val_set)} samples")
    print(f"Test set: {len(test_set)} samples")

    num_pos = sum([1 for _,_,label,_ in train_set.index if label==1])
    num_neg = sum([1 for _,_,label,_ in train_set.index if label==0])
    print(f"TRAIN --- Ictal: {num_pos}, Preictal: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

    num_pos = sum([1 for _,_,label,_ in val_set.index if label==1])
    num_neg = sum([1 for _,_,label,_ in val_set.index if label==0])
    print(f"VAL --- Ictal: {num_pos}, Preictal: {num_neg}, Ratio: {num_pos/num_neg:.6f}")

    num_pos = sum([1 for _,_,label,_ in test_set.index if label==1])
    num_neg = sum([1 for _,_,label,_ in test_set.index if label==0])
    print(f"TEST --- Ictal: {num_pos}, Preictal: {num_neg}, Ratio: {num_pos/num_neg:.6f}")