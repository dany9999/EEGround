
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
import csv
import matplotlib.pyplot as plt


class CHBMITLoader(Dataset):
    def __init__(self, root_dir, segment_files, segment_sec=4, orig_sr=256, target_sr=250):
        self.root_dir = root_dir
        self.segment_files = segment_files
        self.segment_sec = segment_sec
        self.orig_sr = orig_sr
        self.target_sr = target_sr
        self.segment_len = segment_sec * orig_sr
        self.segment_len_down = segment_sec * target_sr
        self.count_no_seizure = 0
        self.count_seizure = 0
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

                # Etichetta = 1 se almeno 2s di sovrapposizione
                if total_overlap > 1:
                    label = 1
                    self.count_seizure += 1
                else:
                    label = 0
                    self.count_no_seizure += 1


                
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

        # Estrai solo il nome del file per usarlo come chiave (es. 'chb01_03.npy')
        file_name = os.path.basename(file_path)

        return (
            torch.tensor(segment_down, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            file_name  # nuovo terzo elemento: nome file
        )




def make_loader(patients_list, root, config, shuffle=False):
    segment_files = []
    for patient in patients_list:
        patient_path = os.path.join(root, patient)
        if os.path.exists(patient_path):
            files = [os.path.join(patient, f) for f in os.listdir(patient_path) if f.endswith(".npy")]
            segment_files.extend(files)

    dataset = CHBMITLoader(
        root_dir=root,
        segment_files=segment_files,
        segment_sec=4,
        orig_sr=config["sampling_rate"],  # 256
        target_sr=250
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        drop_last=shuffle,
        num_workers=config["num_workers"]
    )


# === Test Script ===
def test_dataset():
    config = {
        "sampling_rate": 256,
        "batch_size": 4,
        "num_workers": 0
    }

    root = "CHB-MIT/data"
    patient_id = "chb05"  # esempio
    patient_path = os.path.join(root, patient_id)
    segment_files = [os.path.join(patient_id, f) for f in os.listdir(patient_path) if f.endswith(".npy")]

    dataset = CHBMITLoader(
        root_dir=root,
        segment_files=segment_files,
        segment_sec=4,
        orig_sr=256,
        target_sr=250
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    for i, (x, y) in enumerate(loader):
        print(f"\nBatch {i+1}")
        print(f"Input shape: {x.shape} (B, C, T)")
        print(f"Label: {y.tolist()}")

    
    print(f"Segmenti con label 0 (no seizure): {dataset.count_no_seizure}")
    print(f"Segmenti con label 1 (seizure): {dataset.count_seizure}")

        


if __name__ == "__main__":
    test_dataset()