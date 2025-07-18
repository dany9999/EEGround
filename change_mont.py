import os
import h5py
import numpy as np
from glob import glob
from tqdm import tqdm
from utils import convert_to_bipolar



def process_folder(folder_path):


    print(f"\n Processing: {folder_path}")
    files = sorted(glob(os.path.join(folder_path, "eeg_batch_*.h5")))

    total_sum = None
    total_sq_sum = None
    total_count = 0

    for file_path in tqdm(files):
        with h5py.File(file_path, 'r') as f:
            data = f['signals'][:]  # shape: (batch, channels, samples)

        # Converti in bipolare
        data_bipolar, _ = convert_to_bipolar(data)

        print(f"Processing file: {file_path} | Shape: {data_bipolar.shape}")
        # Update accumulatore
        if total_sum is None:
            total_sum = np.sum(data_bipolar, axis=(0, 2))  # (channels,)
            total_sq_sum = np.sum(data_bipolar ** 2, axis=(0, 2))  # (channels,)
        else:
            total_sum += np.sum(data_bipolar, axis=(0, 2))
            total_sq_sum += np.sum(data_bipolar ** 2, axis=(0, 2))

        total_count += data_bipolar.shape[0] * data_bipolar.shape[2]  # batch * samples

    # Calcolo finale
    mean = total_sum / total_count
    var = (total_sq_sum / total_count) - (mean ** 2)
    std = np.sqrt(np.maximum(var, 1e-6))  # evita numeri negativi

    # Reshape come richiesto: (1, num_channels, 1)
    mean = mean.reshape(1, -1, 1)
    std = std.reshape(1, -1, 1)

    np.save(os.path.join(folder_path, "new_mean.npy"), mean)
    np.save(os.path.join(folder_path, "new_standard_deviation.npy"), std)
    print(f"Saved new_mean.npy and new_standard_deviation.npy in {folder_path}")

if __name__ == "__main__":
    base_path = "/home/inbit/Scrivania/Datasets/TUH/TUAB/REF"
    for subfolder in ["Abnormal", "Normal"]:
        process_folder(os.path.join(base_path, subfolder))