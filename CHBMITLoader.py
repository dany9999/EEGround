
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, DataLoader
# from scipy.signal import resample_poly
# import csv
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader, DistributedSampler

# class CHBMITLoader(Dataset):
#     def __init__(self, root_dir, segment_files, segment_sec=4, orig_sr=256, target_sr=250):
#         self.root_dir = root_dir
#         self.segment_files = segment_files
#         self.segment_sec = segment_sec
#         self.orig_sr = orig_sr
#         self.target_sr = target_sr
#         self.segment_len = segment_sec * orig_sr
#         self.segment_len_down = segment_sec * target_sr
#         self.segments = []  # list of (full_path, start_idx, label)
#         self._prepare_segments()

#     def _prepare_segments(self):
#         for relative_path in self.segment_files:
#             patient_id = relative_path.split("/")[0]
#             file_name = os.path.basename(relative_path)
#             edf_name = os.path.splitext(file_name)[0] + ".edf"
#             full_path = os.path.join(self.root_dir, relative_path)

#             # Leggi intervallo di seizure (in secondi)
#             seizure_ranges = self._get_seizure_ranges(patient_id, edf_name)

#             # Carica file per determinare numero segmenti
#             data = np.load(full_path)
#             total_len = data.shape[1]
#             total_segments = total_len // self.segment_len

#             for i in range(total_segments):
#                 start_sec = i * self.segment_sec
#                 end_sec = start_sec + self.segment_sec

#                 # Calcola durata sovrapposizione
#                 overlaps = [
#                     max(0, min(end_sec, seizure_end) - max(start_sec, seizure_start))
#                     for (seizure_start, seizure_end) in seizure_ranges
#                 ]
#                 total_overlap = sum(overlaps)

#                 # Etichetta = 1 se almeno 1s di sovrapposizione
#                 if total_overlap > 1:
#                     label = 1
                    
#                 else:
#                     label = 0
                    


                
#                 self.segments.append((full_path, i * self.segment_len, label))



#     def _get_seizure_ranges(self, patient_id, edf_file):
#         """
#         Restituisce una lista di tuple [(start_sec, end_sec), ...]
#         """
#         metadata_path = os.path.join("../../Datasets/chb_mit/GT", f"{patient_id}.csv")
#         if not os.path.exists(metadata_path):
#             raise FileNotFoundError(f"CSV metadata non trovato: {metadata_path}")

#         seizure_ranges = []
#         with open(metadata_path, newline='') as f:
#             reader = csv.DictReader(f, delimiter=";")
#             for row in reader:
#                 if row["Name of file"].strip() == edf_file:
#                     if row["class_name"].strip().lower() == "seizure":
#                         start_values = row["Start (sec)"].strip().split("-")
#                         end_values = row["End (sec)"].strip().split("-")

#                         # Assicuriamoci che siano lunghezze compatibili
#                         if len(start_values) != len(end_values):
#                             raise ValueError(f"Numero di start e end diversi in riga: {row}")

#                         for start_str, end_str in zip(start_values, end_values):
#                             try:
#                                 start = int(start_str)
#                                 end = int(end_str)
#                                 seizure_ranges.append((start, end))
#                             except ValueError:
#                                 print(f"Errore di conversione: start={start_str}, end={end_str} nella riga: {row}")
#         return seizure_ranges

#     def __len__(self):
#         return len(self.segments)
    

#     def __getitem__(self, idx):
#         file_path, start_idx, label = self.segments[idx]
#         data = np.load(file_path)  # shape: (21, N)
#         segment = data[:, start_idx : start_idx + self.segment_len]  # (21, 1024)

#         # Downsample a 250 Hz
#         segment_down = resample_poly(segment, self.target_sr, self.orig_sr, axis=1)
#         valid_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # per testare
#         segment_down = segment_down[valid_channels]

#         # Estrai solo il nome del file per usarlo come chiave (es. 'chb01_03.npy')
#         file_name = os.path.basename(file_path)

#         return {
#             "x": torch.tensor(segment_down, dtype=torch.float32),
#             "y": torch.tensor(label, dtype=torch.float32),
#             "file": file_name
#         }


# def has_seizure_segment(patient_id, file_name_no_ext):
#     """
#     Controlla se il file .edf associato ha almeno una crisi, basandosi sul campo 'Numb of seizures'.
#     """
#     csv_path = os.path.join("../../Datasets/chb_mit/GT", f"{patient_id}.csv")
#     if not os.path.exists(csv_path):
#         return False

#     with open(csv_path, newline='') as f:
#         reader = csv.DictReader(f, delimiter=";")
#         for row in reader:
#             if row["Name of file"].strip() == file_name_no_ext + ".edf":
#                 try:
#                     return int(row["Numb of seizures"]) > 0
#                 except (KeyError, ValueError):
#                     return row.get("class_name", "").strip().lower() == "seizure"
#     return False

# # def make_loader(patients_list, root, config, shuffle=False):
# #     segment_files = []
# #     for patient in patients_list:
# #         patient_path = os.path.join(root, patient)
# #         if os.path.exists(patient_path):
# #             for f in os.listdir(patient_path):
# #                 if not f.endswith(".npy"):
# #                     continue
# #                 file_name_no_ext = os.path.splitext(f)[0]  # es. chb02_01
# #                 if has_seizure_segment(patient, file_name_no_ext):
# #                     segment_files.append(os.path.join(patient, f))

# #     dataset = CHBMITLoader(
# #         root_dir=root,
# #         segment_files=segment_files,
# #         segment_sec=4,
# #         orig_sr=256,
# #         target_sr=250
# #     )
# #     return DataLoader(
# #         dataset,
# #         batch_size=config["batch_size"],
# #         shuffle=shuffle,
# #         drop_last=shuffle,
# #         num_workers=config["num_workers"]
# #     )




# def make_loader(patients_list, root, config, shuffle=False, is_ddp=False, rank=0, world_size=1):
#     """
#     Crea un DataLoader, con supporto opzionale per DistributedSampler (DDP).
    
#     Args:
#         patients_list (list): Lista dei pazienti da includere
#         root (str): Path alla directory dei dati
#         config (dict): Configurazione generale
#         shuffle (bool): Se fare lo shuffle dei dati
#         is_ddp (bool): Se usare DistributedSampler per DDP
#         rank (int): Rank del processo corrente (necessario per DDP)
#         world_size (int): Numero totale di processi DDP
#     """
#     segment_files = []
#     for patient in patients_list:
#         patient_path = os.path.join(root, patient)
#         if os.path.exists(patient_path):
#             for f in os.listdir(patient_path):
#                 if not f.endswith(".npy"):
#                     continue
#                 file_name_no_ext = os.path.splitext(f)[0]
#                 if has_seizure_segment(patient, file_name_no_ext):
#                     segment_files.append(os.path.join(patient, f))

#     dataset = CHBMITLoader(
#         root_dir=root,
#         segment_files=segment_files,
#         segment_sec=4,
#         orig_sr=256,
#         target_sr=250
#     )

#     if is_ddp:
#         sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
#         loader = DataLoader(
#             dataset,
#             batch_size=config["batch_size"],
#             sampler=sampler,
#             num_workers=config["num_workers"],
#             pin_memory=True,
#             drop_last=True,      # Scarta l'ultimo batch se è più piccolo del batch_size
#         )
#     else:
#         loader = DataLoader(
#             dataset,
#             batch_size=config["batch_size"],
#             num_workers=config["num_workers"],
#             shuffle=shuffle,        # Mescola i dati ogni epoca
#             drop_last=True,      # Scarta l'ultimo batch se è più piccolo del batch_size
#         )

#     return loader


# import os
# import torch
# import h5py
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader, DistributedSampler

# class CHBMITAllSegmentsLabeledDataset(Dataset):
#     def __init__(self, patient_ids, data_dir, gt_dir,
#                  segment_duration_sec=4, transform=None):
#         self.samples = []
#         self.labels = []
#         self.transform = transform

#         for patient in patient_ids:
#             patient_folder = os.path.join(data_dir, patient)
#             gt_file = os.path.join(gt_dir, f"{patient}.csv")
#             if not os.path.exists(gt_file):
#                 continue

#             # parsing ground truth
#             gt_df = pd.read_csv(gt_file, sep=';', engine='python')
#             seizure_map = {}
#             for _, row in gt_df.iterrows():
#                 edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
#                 if int(row["Numb of seizures"]) > 0:
#                     starts = [float(s) for s in str(row["Start (sec)"]).split(',')]
#                     ends = [float(e) for e in str(row["End (sec)"]).split(',')]
#                     seizure_map[edf_base] = list(zip(starts, ends))

#             for fname in sorted(os.listdir(patient_folder)):
#                 print(f"Patient {patient}: found {len(seizure_map)} seizure files")
#                 print(f" --> Samples accumulated: {len(self.samples)}")
#                 if not fname.endswith(".h5"):
#                     continue

#                 edf_base = fname.replace(".h5", "")
#                 fpath = os.path.join(patient_folder, fname)


#                 # if edf_base not in seizure_map:
#                 #    continue

#                 with h5py.File(fpath, 'r') as f:
#                     x = f['signals'][:]  # shape: (segments, ch, time)

#                 for i in range(x.shape[0]):
#                     seg_start = i * segment_duration_sec
#                     seg_end = seg_start + segment_duration_sec

#                     label = 0
#                     for (st, en) in seizure_map[edf_base]:
#                         if not (seg_end <= st or seg_start >= en):
#                             label = 1
#                             break

#                     self.samples.append(torch.tensor(x[i], dtype=torch.float32))
#                     self.labels.append(label)

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         x = self.samples[idx]
#         y = self.labels[idx]
#         if self.transform:
#             x = self.transform(x)
#         return {"x": x, "y": torch.tensor(y, dtype=torch.long)}

# def make_loader(patient_ids, dataset_path,GT_path, config,
#                 shuffle=True, is_ddp=False, rank=0, world_size=1):
#     dataset = CHBMITAllSegmentsLabeledDataset(
#         patient_ids=patient_ids,
#         data_dir=os.path.join(dataset_path, "bipolar_data"),
#         gt_dir=os.path.join(GT_path, "GT"),
#         segment_duration_sec= 4,
#         transform=None
#     )

#     if is_ddp:
#         sampler = DistributedSampler(dataset,
#                                      num_replicas=world_size,
#                                      rank=rank,
#                                      shuffle=shuffle)
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

    
#     patient_ids = ["chb01", "chb02"]  # Replace with actual patient IDs
#     dataset_path = "../../Datasets/Bipolar/chb_mit"
#     GT_path = "../../Datasets/chb_mit"
#     config = {
#         "batch_size": 32,
#         "num_workers": 4,
#         "segment_duration_sec": 4
#     }
    
#     loader = make_loader(patient_ids, dataset_path, GT_path, config, shuffle=True)

#     print(f"Number of batches: {len(loader)}")
#     print(f"Total samples in dataset: {len(loader.dataset)}")

#     # Visualizza le prime etichette
#     for i, batch in enumerate(loader):
#         print(f"Batch {i+1}: x shape = {batch['x'].shape}, y = {batch['y']}")
#         if i == 2: break



import os
import torch
import h5py
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from utils import load_config

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
                print(f"⚠️ GT not found for {patient}, skipping")
                continue

            # parsing ground truth CSV
            gt_df = pd.read_csv(gt_file, sep=';', engine='python')
          
            seizure_map = {}
            for _, row in gt_df.iterrows():
                edf_base = os.path.splitext(os.path.basename(row["Name of file"]))[0]
                if isinstance(row["class_name"], str) and "no seizure" in row["class_name"].lower():
                    seizure_map[edf_base] = []
                else:
                    starts = str(row["Start (sec)"]).split(',') if pd.notna(row["Start (sec)"]) else []
                    ends   = str(row["End (sec)"]).split(',') if pd.notna(row["End (sec)"]) else []
                    starts = [float(s) for s in starts if s not in ["", "0"]]
                    ends   = [float(e) for e in ends if s not in ["", "0"]]
                    seizure_map[edf_base] = list(zip(starts, ends))

            print("Parsed GT:")
            print(gt_df.head())
            print("Seizure map keys:", seizure_map.keys())

            # Scorri gli h5 nella cartella del paziente
            for fname in sorted(os.listdir(patient_folder)):
                if not fname.endswith(".h5"):
                    continue
                edf_base = fname.replace(".h5", "")
                fpath = os.path.join(patient_folder, fname)

                with h5py.File(fpath, 'r') as f:
                    n_segments = f['signals'].shape[0]

                intervals = seizure_map.get(edf_base, [])

                # crea un indice di segmenti con etichetta
                for i in range(n_segments):
                    seg_start = i * self.segment_duration_sec
                    seg_end   = seg_start + self.segment_duration_sec

                    label = 0
                    for (st, en) in intervals:
                        if not (seg_end <= st or seg_start >= en):
                            label = 1
                            break

                    self.index.append((fpath, i, label, edf_base))

            print(f"[{patient}] -> {len(self.index)} total segments accumulated")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fpath, i, label, file_id = self.index[idx]
        with h5py.File(fpath, 'r') as f:
            x = f['signals'][i]  # (channels, time)
        x = torch.tensor(x, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return {"x": x, "y": torch.tensor(label, dtype=torch.long), "file": file_id}
    
def make_loader(patient_ids, dataset_path, gt_path, config,
                shuffle=True, is_ddp=False, rank=0, world_size=1):
    dataset = CHBMITAllSegmentsLabeledDataset(
        patient_ids=patient_ids,
        data_dir=dataset_path,
        gt_dir=gt_path,
        segment_duration_sec=config.get("segment_duration_sec", 4),
        transform=None
    )

    if is_ddp:
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
    # Example usage
    patient_ids = ["chb01", "chb02"]
    config = load_config("configs/finetuning.yml")

    dataset_path = config["dataset_path"]
    gt_path = config["gt_path"]

    
    loader = make_loader(patient_ids, dataset_path, gt_path, config, shuffle=True)
    
    print(f"Number of batches: {len(loader)}")
    print(f"Total samples in dataset: {len(loader.dataset)}")
    # Visualizza le prime etichette
    for i, batch in enumerate(loader):
        print(f"Batch {i+1}: x shape = {batch['x'].shape}, y = {batch['y']}, file = {batch['file']}")
        if i == 2: break

