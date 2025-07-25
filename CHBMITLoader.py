
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
import csv
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class CHBMITLoader(Dataset):
    def __init__(self, root_dir, segment_files, segment_sec=4, orig_sr=256, target_sr=250):
        self.root_dir = root_dir
        self.segment_files = segment_files
        self.segment_sec = segment_sec
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.segment_len = segment_sec * orig_sr
        self.segment_len_down = segment_sec * target_sr
        self.segments = []  # list of (full_path, start_idx, label)
        self._prepare_segments()

    def _prepare_segments(self):
        for relative_path in self.segment_files:
            patient_id = relative_path.split("/")[0]
            file_name = os.path.basename(relative_path)
            edf_name = os.path.splitext(file_name)[0] + ".edf"
            full_path = os.path.join(self.root_dir, relative_path)

            # Leggi intervallo di seizure (in secondi)
            seizure_ranges = self._get_seizure_ranges(patient_id, edf_name)

            # Carica file per determinare numero segmenti
            data = np.load(full_path)
            total_len = data.shape[1]
            total_segments = total_len // self.segment_len

            for i in range(total_segments):
                start_sec = i * self.segment_sec
                end_sec = start_sec + self.segment_sec

                # Calcola durata sovrapposizione
                overlaps = [
                    max(0, min(end_sec, seizure_end) - max(start_sec, seizure_start))
                    for (seizure_start, seizure_end) in seizure_ranges
                ]
                total_overlap = sum(overlaps)

                # Etichetta = 1 se almeno 1s di sovrapposizione
                if total_overlap > 1:
                    label = 1
                    
                else:
                    label = 0
                    


                
                self.segments.append((full_path, i * self.segment_len, label))



    def _get_seizure_ranges(self, patient_id, edf_file):
        """
        Restituisce una lista di tuple [(start_sec, end_sec), ...]
        """
        metadata_path = os.path.join("../../Datasets/chb_mit/GT", f"{patient_id}.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"CSV metadata non trovato: {metadata_path}")

        seizure_ranges = []
        with open(metadata_path, newline='') as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row["Name of file"].strip() == edf_file:
                    if row["class_name"].strip().lower() == "seizure":
                        start_values = row["Start (sec)"].strip().split("-")
                        end_values = row["End (sec)"].strip().split("-")

                        # Assicuriamoci che siano lunghezze compatibili
                        if len(start_values) != len(end_values):
                            raise ValueError(f"Numero di start e end diversi in riga: {row}")

                        for start_str, end_str in zip(start_values, end_values):
                            try:
                                start = int(start_str)
                                end = int(end_str)
                                seizure_ranges.append((start, end))
                            except ValueError:
                                print(f"Errore di conversione: start={start_str}, end={end_str} nella riga: {row}")
        return seizure_ranges

    def __len__(self):
        return len(self.segments)
    

    def __getitem__(self, idx):
        file_path, start_idx, label = self.segments[idx]
        data = np.load(file_path)  # shape: (21, N)
        segment = data[:, start_idx : start_idx + self.segment_len]  # (21, 1024)

        # Downsample a 250 Hz
        segment_down = resample_poly(segment, self.target_sr, self.orig_sr, axis=1)
        valid_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]  # per testare
        segment_down = segment_down[valid_channels]

        # Estrai solo il nome del file per usarlo come chiave (es. 'chb01_03.npy')
        file_name = os.path.basename(file_path)

        return {
            "x": torch.tensor(segment_down, dtype=torch.float32),
            "y": torch.tensor(label, dtype=torch.float32),
            "file": file_name
        }


def has_seizure_segment(patient_id, file_name_no_ext):
    """
    Controlla se il file .edf associato ha almeno una crisi, basandosi sul campo 'Numb of seizures'.
    """
    csv_path = os.path.join("../../Datasets/chb_mit/GT", f"{patient_id}.csv")
    if not os.path.exists(csv_path):
        return False

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            if row["Name of file"].strip() == file_name_no_ext + ".edf":
                try:
                    return int(row["Numb of seizures"]) > 0
                except (KeyError, ValueError):
                    return row.get("class_name", "").strip().lower() == "seizure"
    return False

# def make_loader(patients_list, root, config, shuffle=False):
#     segment_files = []
#     for patient in patients_list:
#         patient_path = os.path.join(root, patient)
#         if os.path.exists(patient_path):
#             for f in os.listdir(patient_path):
#                 if not f.endswith(".npy"):
#                     continue
#                 file_name_no_ext = os.path.splitext(f)[0]  # es. chb02_01
#                 if has_seizure_segment(patient, file_name_no_ext):
#                     segment_files.append(os.path.join(patient, f))

#     dataset = CHBMITLoader(
#         root_dir=root,
#         segment_files=segment_files,
#         segment_sec=4,
#         orig_sr=256,
#         target_sr=250
#     )
#     return DataLoader(
#         dataset,
#         batch_size=config["batch_size"],
#         shuffle=shuffle,
#         drop_last=shuffle,
#         num_workers=config["num_workers"]
#     )




def make_loader(patients_list, root, config, shuffle=False, is_ddp=False, rank=0, world_size=1):
    """
    Crea un DataLoader, con supporto opzionale per DistributedSampler (DDP).
    
    Args:
        patients_list (list): Lista dei pazienti da includere
        root (str): Path alla directory dei dati
        config (dict): Configurazione generale
        shuffle (bool): Se fare lo shuffle dei dati
        is_ddp (bool): Se usare DistributedSampler per DDP
        rank (int): Rank del processo corrente (necessario per DDP)
        world_size (int): Numero totale di processi DDP
    """
    segment_files = []
    for patient in patients_list:
        patient_path = os.path.join(root, patient)
        if os.path.exists(patient_path):
            for f in os.listdir(patient_path):
                if not f.endswith(".npy"):
                    continue
                file_name_no_ext = os.path.splitext(f)[0]
                if has_seizure_segment(patient, file_name_no_ext):
                    segment_files.append(os.path.join(patient, f))

    dataset = CHBMITLoader(
        root_dir=root,
        segment_files=segment_files,
        segment_sec=4,
        orig_sr=256,
        target_sr=250
    )

    if is_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            sampler=sampler,
            num_workers=config["num_workers"],
            pin_memory=True,
            drop_last=True,      # Scarta l'ultimo batch se è più piccolo del batch_size
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
            shuffle=shuffle,        # Mescola i dati ogni epoca
            drop_last=True,      # Scarta l'ultimo batch se è più piccolo del batch_size
        )

    return loader


