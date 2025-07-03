import os
import json
import random
import yaml
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from model.SelfSupervisedPretrain import UnsupervisedPretrain

# ==== Utils ====

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

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


# ==== Mean/Std Loader ====

class MeanStdLoader:
    def __init__(self):
        self.cache = {}

    def get_mean_std_for_file(self, file_path, device):
        file_path = os.path.abspath(file_path)
        folder = os.path.dirname(file_path)

        if folder not in self.cache:
            mean_path = os.path.join(folder, "mean.npy")
            std_path = os.path.join(folder, "standard_deviation.npy")

            mean_all = np.load(mean_path).squeeze()
            std_all = np.load(std_path).squeeze()

            self.cache[folder] = {
                "mean_all": mean_all,
                "std_all": std_all
            }

        mean = torch.tensor(self.cache[folder]["mean_all"], dtype=torch.float32).to(device)
        std = torch.tensor(self.cache[folder]["std_all"], dtype=torch.float32).to(device)
        return mean, std

# ==== Dataset ====

class EEGDataset(Dataset):
    def __init__(self, data_array):
        self.data = torch.tensor(data_array, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==== Training ====

def train_one_file(model, optimizer, file_path, batch_size, device, writer, global_step, mean_std_loader):
    with h5py.File(file_path, 'r') as f:
        data = f["signals"][:]

    mean, std = mean_std_loader.get_mean_std_for_file(file_path, device)
    mean_exp = mean.view(1, -1, 1)
    std_exp = std.view(1, -1, 1)

    dataset = EEGDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    running_loss = 0.0

    for batch_raw in dataloader:
        batch_raw = batch_raw.to(device)
        batch_norm = (batch_raw - mean_exp) / std_exp

        optimizer.zero_grad()
        output = model(batch_norm)
        loss = F.mse_loss(output, batch_raw)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        writer.add_scalar("BatchLoss/Train", loss.item(), global_step)
        global_step += 1
    
    avg_loss = running_loss / len(dataloader)

    return avg_loss, global_step

def validate_one_file(model, file_path, batch_size, device, writer, global_step_val, mean_std_loader):
    with h5py.File(file_path, 'r') as f:
        data = f["signals"][:]

    mean, std = mean_std_loader.get_mean_std_for_file(file_path, device)
    mean_exp = mean.view(1, -1, 1)
    std_exp = std.view(1, -1, 1)

    dataset = EEGDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_raw in dataloader:
            batch_raw = batch_raw.to(device)
            batch_norm = (batch_raw - mean_exp) / std_exp

            output = model(batch_norm)
            loss = F.mse_loss(output, batch_raw)

            running_loss += loss.item()
            writer.add_scalar("BatchLoss/Val", loss.item(), global_step_val)
            global_step_val += 1
    avg_loss = running_loss / len(dataloader)
    return avg_loss, global_step_val

# ==== Main Training Loop ====

def train_model(config):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("Collecting h5 files...")
    dataset_path = os.path.abspath(config["dataset_path"])
    all_files = collect_h5_files(dataset_path)
    train_files = all_files[:int(0.7 * len(all_files))]
    val_files = all_files[int(0.7 * len(all_files)):]
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    log_dir = config.get("log_dir", "./logs/pretrain")
    os.makedirs(log_dir, exist_ok=True)



    model = UnsupervisedPretrain(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_channels=config["n_channels"],
        mask_ratio=config["mask_ratio"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    writer = SummaryWriter(log_dir=log_dir)
    mean_std_loader = MeanStdLoader()

    start_epoch = 0
    global_step = 0
    global_step_val = 0

    checkpoints = sorted(glob(os.path.join(log_dir, "model_epoch_*.pt")))
    if checkpoints:
        latest_ckpt = checkpoints[-1]
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', 0)
        global_step_val = checkpoint.get('global_step_val', 0)

    for epoch in range(start_epoch, config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")


        train_losses = []
        for f in tqdm(train_files, desc="Training"):
            loss, global_step = train_one_file(model, optimizer, f, config["batch_size"], device, writer, global_step, mean_std_loader)
            train_losses.append(loss)

        val_losses = []
        for f in tqdm(val_files, desc="Validation"):
            loss, global_step_val = validate_one_file(model, f, config["batch_size"], device, writer, global_step_val, mean_std_loader)
            val_losses.append(loss)

        train_loss = sum(train_losses) / len(train_losses) 
        val_loss = sum(val_losses) / len(val_losses) 

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", val_loss, epoch + 1)

        if (epoch + 1) % config.get("save_every", 1) == 0:
            save_path = os.path.join(log_dir, f"model_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'global_step_val': global_step_val
            }, save_path)
            print(f"Saved model checkpoint to {save_path}")

    writer.close()

# ==== Run ====

if __name__ == "__main__":
    config = load_config("configs/pretraining.yml")
    train_model(config)


