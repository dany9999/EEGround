

import pyedflib
import os
import numpy as np
from tqdm import tqdm


def make_dataset(patient_folders, processed_root):
    # Root path del dataset
    dataset_root = "./CHB-MIT/"
    os.makedirs(processed_root, exist_ok=True)

    # Parametri noti
    segment_duration = 10  # secondi
    sampling_rate = 256  # fissa a 256 Hz (CHB-MIT)
    segment_length = segment_duration * sampling_rate

    # Liste per unione globale
    all_segments = []
    all_sources = []  # per tracciare da dove viene ogni segmento

    # Scorri solo le cartelle specificate
    for patient_folder in patient_folders:
        patient_path = os.path.join(dataset_root, patient_folder)
        if not os.path.isdir(patient_path):
            continue

        print(f"\nProcessing paziente: {patient_folder}")
        
        # Scorri tutti i file EDF del paziente
        for file_name in sorted(os.listdir(patient_path)):
            if not file_name.endswith(".edf"):
                continue

            file_path = os.path.join(patient_path, file_name)

            try:
                with pyedflib.EdfReader(file_path) as edf_file:
                    n_channels = edf_file.signals_in_file
                    total_samples = edf_file.getNSamples()[0]
                    n_segments = total_samples // segment_length

                    # Scegli i 18 canali da usare
                    selected_channels = list(range(18))  # oppure escludi canali specifici

                    # Leggi solo i canali selezionati
                    signals = np.array([edf_file.readSignal(ch) for ch in selected_channels])

                    for idx in range(n_segments):
                        start = idx * segment_length
                        end = start + segment_length
                        segment = signals[:, start:end]
                        if segment.shape[1] == segment_length:
                            all_segments.append(segment)
                            all_sources.append({
                                "patient": patient_folder,
                                "file": file_name,
                                "segment_index": idx
                            })

            except Exception as e:
                print(f"Errore con {file_name}: {e}")

    # Converti in array numpy
    all_segments_array = np.array(all_segments)  # shape: (N, channels, 2560)
    print(f"\nTotale segmenti: {all_segments_array.shape[0]}")

    # Salva
    np.save(os.path.join(processed_root, "all_segments.npy"), all_segments_array)

    # Salva le info di origine per riferimento
    import json
    with open(os.path.join(processed_root, "all_sources.json"), "w") as f:
        json.dump(all_sources, f, indent=2)

    print("Segmenti salvati con successo.")


def make_dataset_pretrain(patient_folders, processed_root):
    " devo splittare TUH in train e validation set, successivamente"

    pass

def make_dataset_finetuning(patient_folders, processed_root):
    pass


def main():
    # Training set: chb01, chb02
    make_dataset(["chb01", "chb02"], "./CHB-MIT/train_numpy")
    # Validation set: chb03
    make_dataset(["chb03"], "./CHB-MIT/validation_numpy")

if __name__ == "__main__":
    main()


