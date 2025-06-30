
import numpy as np
import os

def check_mean_std_shapes(folder_path):
    mean_path = os.path.join(folder_path, "mean.npy")
    std_path = os.path.join(folder_path, "standard_deviation.npy")

    if os.path.exists(mean_path):
        mean = np.load(mean_path)
        print(f"mean.npy shape: {mean.shape}")
        print(f"mean.npy sample data:\n{mean if mean.size < 10 else mean[:10]}")
    else:
        print("mean.npy not found!")

    if os.path.exists(std_path):
        std = np.load(std_path)
        print(f"standard_deviation.npy shape: {std.shape}")
        print(f"standard_deviation.npy sample data:\n{std if std.size < 10 else std[:10]}")
    else:
        print("standard_deviation.npy not found!")

if __name__ == "__main__":
    folder = "/home/inbit/Scrivania/Datasets/TUH/TUAB/Abnormal/REF"  # cambia con la tua cartella
    check_mean_std_shapes(folder)