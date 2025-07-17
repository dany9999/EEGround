

import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm
from model.SelfSupervisedPretrainEMB import UnsupervisedPretrain
from utils import MeanStdLoader, EEGDataset, load_config, collect_h5_files
import random
from utils import convert_to_bipolar


# ==== Training ====

def train_one_file(model, optimizer, file_path, batch_size, device, writer, global_step, mean_std_loader):
    with h5py.File(file_path, 'r') as f:
        data = f["signals"][:]

    data = convert_to_bipolar(data)
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

        emb, mask, _, pred_emb = model(batch_norm)
        loss = F.mse_loss(pred_emb[mask], emb[mask])
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
            
            
            emb, mask, _, pred_emb = model(batch_norm)
            loss = F.mse_loss(pred_emb[mask], emb[mask])

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

   # ---- Fissa il seed per randomizzazione coerente ----
    random.seed(246)  

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    # ---- Split fisso: carica o crea ----
    split_path = os.path.join(log_dir, "file_split.npy")
    if os.path.exists(split_path):
        print(f"Loading train/val split from {split_path}")
        split = np.load(split_path, allow_pickle=True).item()
        train_files, val_files = split["train"], split["val"]
    else:
        print("Shuffling and creating train/val split...")
        random.shuffle(all_files)
        train_files = all_files[:int(0.7 * len(all_files))]
        val_files = all_files[int(0.7 * len(all_files)):]
        np.save(split_path, {"train": train_files, "val": val_files})
        print(f"Saved train/val split to {split_path}")

    print(f"Training files: {len(train_files)}, Validation files: {len(val_files)}")

    log_dir = config["log_dir"]
    os.makedirs(log_dir, exist_ok=True)



    model = UnsupervisedPretrain(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_channels=config["n_channels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        mask_ratio=config["mask_ratio"]
    )
    model = torch.nn.DataParallel(model).to(device)

    

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


