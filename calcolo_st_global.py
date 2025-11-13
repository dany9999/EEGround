import h5py
import numpy as np
from glob import glob
import torch
from tqdm import tqdm

# ====== CONFIG ==========================================================
global_mean = np.load("global_mean.npy")   # shape (C,)
global_std  = np.load("global_std.npy")    # shape (C,)
train_files = np.load("log/pretrain-new/file_split.npy", allow_pickle=True).item()["train"]

global_mean_t = torch.tensor(global_mean, dtype=torch.float32)
global_std_t  = torch.tensor(global_std, dtype=torch.float32)

# ============ ACCUMULATOR ==============================================
sum_norm = torch.zeros_like(global_mean_t)    # sum over channels
sum_norm2 = torch.zeros_like(global_mean_t)   # sum of squares
total_points = 0

# =======================================================================
print("\nComputing MEAN/STDEV AFTER NORMALIZATION on ALL TRAIN SET\n")

for fpath in tqdm(train_files):
    with h5py.File(fpath, "r") as f:
        data = torch.tensor(f["signals"][:], dtype=torch.float32)   # (N,C,T)

    # normalize
    data_norm = (data - global_mean_t.view(1,-1,1)) / global_std_t.view(1,-1,1)

    # sum over N and T
    sum_norm  += data_norm.sum(dim=(0,2))
    sum_norm2 += (data_norm**2).sum(dim=(0,2))

    total_points += data_norm.shape[0] * data_norm.shape[2]

# ======= final ==========================================================
mean_post = sum_norm / total_points
var_post  = (sum_norm2 / total_points) - mean_post**2
std_post  = torch.sqrt(var_post + 1e-8)

print("\n=== FINAL TRAIN STATISTICS AFTER NORMALIZATION ===")
print("Mean per channel:", mean_post)
print("Std per channel :", std_post)
print("\nGLOBAL mean :", mean_post.mean().item())
print("GLOBAL std  :", std_post.mean().item())