import os
import random
import yaml
import h5py
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from model.SelfSupervisedPretrain import UnsupervisedPretrain  # verifica il path

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
        # Scorro Normal e Abnormal (se presenti)
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

# ==== Dataset e Dataloader ====

class EEGDataset(Dataset):
    def __init__(self, data_array):
        self.data = data_array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)

def train_one_file(model, optimizer, file_path, batch_size, device):
    with h5py.File(file_path, 'r') as f:
        data = f["signals"][:]
    dataset = EEGDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model.train()
    running_loss = 0.0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()
        emb_clean_all, reconstruction = model(batch)
        loss = F.mse_loss(emb_clean_all, reconstruction)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(dataloader)

def validate_one_file(model, file_path, batch_size, device):
    with h5py.File(file_path, 'r') as f:
        data = f["signals"][:]
    dataset = EEGDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb_clean_all, reconstruction = model(batch)
            loss = F.mse_loss(emb_clean_all, reconstruction)
            running_loss += loss.item()

    return running_loss / len(dataloader)

# ==== Main Training Loop ====

def train_model(config):
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    print("Collecting h5 files...")

    dataset_path = os.path.abspath(os.path.join("..", "..", "Datasets/TUH"))
    all_files = collect_h5_files(dataset_path)
    train_files, val_files = split_dataset(all_files, train_ratio=0.7)
    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    model = UnsupervisedPretrain(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_channels=config["n_channels"],
        mask_ratio=config["mask_ratio"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr= float(config["lr"]), weight_decay= float(config["weight_decay"]))

    log_dir = config.get("log_dir", "./logs/pretrain")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        train_losses = []
        for path in tqdm(train_files, desc="Training"):
            loss = train_one_file(model, optimizer, path, config["batch_size"], device)
            train_losses.append(loss)

        val_losses = []
        for path in tqdm(val_files, desc="Validation"):
            loss = validate_one_file(model, path, config["batch_size"], device)
            val_losses.append(loss)

        train_loss = sum(train_losses) / len(train_losses)
        val_loss = sum(val_losses) / len(val_losses)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", val_loss, epoch + 1)

        if (epoch + 1) % config["save_every"] == 0 or (epoch + 1) == config["epochs"]:
            save_path = os.path.join(log_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved model checkpoint to {save_path}")

    writer.close()

# ==== Entry Point ====

if __name__ == "__main__":
    config = load_config("configs/pretraining.yml")
    train_model(config)