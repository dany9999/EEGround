import os
import numpy as np

def inspect_numpy_files(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".npy")]
    print(f"Trovati {len(files)} file .npy in {folder_path}\n")
    for file in sorted(files):
        file_path = os.path.join(folder_path, file)
        try:
            data = np.load(file_path)
            print(f"{file}: shape = {data.shape}, dtype = {data.dtype}")
        except Exception as e:
            print(f"{file}: errore nel caricamento - {e}")

if __name__ == "__main__":
    dataset_path = os.path.abspath("/Scrivania/Datasets/Bipolar/chb_mit/bipolar_data/chb01")
    inspect_numpy_files(dataset_path)