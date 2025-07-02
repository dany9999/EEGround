
import os
import glob
import random

def collect_h5_files(root_dir):
    all_files = []
    subdatasets = ['TUAB', 'TUEP', 'TUEV', 'TUSZ']
    for sub in subdatasets:
        sub_path = os.path.join(root_dir, sub)
        if not os.path.exists(sub_path):
            continue
        for condition in ['Normal', 'Abnormal']:
            cond_path = os.path.join(sub_path, condition, 'REF')
            if os.path.exists(cond_path):
                files = glob(os.path.join(cond_path, "*.h5"))
                files = [f for f in files if not f.endswith(('mean.npy', 'standard_deviation.npy'))]
                all_files.extend(files)
    return sorted(all_files)

def split_dataset(files, train_ratio=0.7, seed=42):
    random.seed(seed)
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)
    return files[:split_idx], files[split_idx:]



dataset_path = os.path.abspath(os.path.join("..", "..", "Datasets/TUH"))
all_files = collect_h5_files(dataset_path)
train_files, val_files = split_dataset(all_files, train_ratio=0.7)
print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")
print("Sample training files:", train_files)
print("Sample validation files:", val_files)

