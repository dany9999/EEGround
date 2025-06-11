import pickle
import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp

root = "CHB-MIT/clean_signals"
out = "CHB-MIT/clean_segments"


channels = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
]
SAMPLING_RATE = 256


def sub_to_segments(folder, out_folder):
    print(f"Processing {folder}...")
    for f in tqdm(os.listdir(os.path.join(root, folder))):
        print(f"Processing {folder}/{f}...")
        record = pickle.load(open(os.path.join(root, folder, f), "rb"))

        signal = []
        for channel in channels:
            if channel in record:
                signal.append(record[channel])
            else:
                raise ValueError(f"Channel {channel} not found in record {f}")

        signal = np.array(signal)

        seizure_times = record.get("metadata", {}).get("times", [])

        # segmenta il segnale in finestre da 10 secondi
        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i : i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                label = 0
                for start, end in seizure_times:
                    if not (end < i or start > i + 10 * SAMPLING_RATE):
                        label = 1
                        break
                # salva il segmento
                with open(os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"), "wb") as handle:
                    pickle.dump({"X": segment, "y": label}, handle)

        # segmenti aggiuntivi intorno ai tempi di crisi
        for idx, (start_time, end_time) in enumerate(seizure_times):
            for i in range(
                max(0, start_time - SAMPLING_RATE),
                min(end_time + SAMPLING_RATE, signal.shape[1]),
                5 * SAMPLING_RATE,
            ):
                segment = signal[:, i : i + 10 * SAMPLING_RATE]
                if segment.shape[1] == 10 * SAMPLING_RATE:
                    label = 1
                    with open(os.path.join(out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"), "wb") as handle:
                        pickle.dump({"X": segment, "y": label}, handle)


if __name__ == "__main__":
    folders = os.listdir(root)
    out_folders = []

    for folder in folders:
        out_folder = os.path.join(out, folder)  # es: clean_segments/chb01
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        out_folders.append(out_folder)

    with mp.Pool(mp.cpu_count()) as pool:
        pool.starmap(sub_to_segments, zip(folders, out_folders))