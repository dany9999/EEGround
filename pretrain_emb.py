

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import re

# ==== CALCOLO GLOBAL MEAN & STD SU TRAIN ====

def compute_global_mean_std(file_list):
    print("\nComputing GLOBAL mean/std over training set...")
    sum_x = None
    sum_x2 = None
    total_samples = 0

    for fpath in tqdm(file_list, desc="Mean/Std"):
        with h5py.File(fpath, "r") as f:
            data = f["signals"][:]   # shape: [N, C, T]

        # somma su batch
        x = data  # (N,C,T)
        N = x.shape[0]

        x_sum = x.sum(axis=(0, 2))          # (C,)
        x2_sum = (x ** 2).sum(axis=(0, 2))  # (C,)

        if sum_x is None:
            sum_x = x_sum
            sum_x2 = x2_sum
        else:
            sum_x += x_sum
            sum_x2 += x2_sum

        total_samples += N * x.shape[2]     # totale punti per canale

    mean = sum_x / total_samples
    var = sum_x2 / total_samples - mean ** 2
    std = np.sqrt(var + 1e-8)

    print("\nGLOBAL mean:", mean)
    print("GLOBAL std:", std)
    return mean.astype(np.float32), std.astype(np.float32)



# ==== Training ====

def train_one_file(model, optimizer, file_path, batch_size, device, writer, global_step, mean_std_loader):
    with h5py.File(file_path, 'r') as f:
        data = f["signals"][:]

   
    mean, std = mean_std_loader.get_mean_std()
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

    mean, std  = mean_std_loader.get_mean_std()
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

# save checkpoint function
def save_best_model(model, save_dir, epoch):
    """
    Salva sia il checkpoint completo del modello di pretraining (UnsupervisedPretrain)
    sia solo i pesi dell'encoder BIOTEMB, rinominati per il fine-tuning.
    Gestisce modelli wrappati in nn.DataParallel.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ---  Salva il modello completo ---
    full_ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
    }
    full_path = os.path.join(save_dir, f"best_full_model.pt")
    torch.save(full_ckpt, full_path)
    print(f" Miglior model_state_dict salvato in {full_path}")

    # --- Estrai solo i pesi dell'encoder (BIOTEMB) ---
    raw_state = model.state_dict()
    encoder_state = {}

    for k, v in raw_state.items():
        key = k
        # Rimuovi il prefisso 'module.' se presente
        if key.startswith("module."):
            key = key[len("module."):]
        # Ora prendi tutto ciò che inizia con 'biot.'
        if key.startswith("biot."):
            encoder_state[key] = v

    encoder_path = os.path.join(save_dir, f"best_encoder_only.pt")
    torch.save(encoder_state, encoder_path)
    
    print(f" Encoder salvato in {encoder_path} ({len(encoder_state)} layer)")


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

        # === calcola solo una volta ===
    #global_mean, global_std = compute_global_mean_std(train_files)
    global_mean = np.load("global_mean.npy")
    global_std = np.load("global_std.npy")
    # salvalo per sicurezza
    #np.save( "global_mean.npy", global_mean)
    #np.save( "global_std.npy", global_std)

    mean_std_loader = MeanStdLoader(global_mean, global_std, device)

    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,      # 10 epoche per il primo ciclo
    T_mult=2,    # ogni volta raddoppia la lunghezza del ciclo
    eta_min=1e-6
)
    writer = SummaryWriter(log_dir=log_dir)
   

    start_epoch = 0
    global_step = 0
    global_step_val = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    checkpoints = sorted(glob(os.path.join(log_dir, "checkpoint_epoch_*.pt")))
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

        #scheduler.step(val_loss)
        scheduler.step()

        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch + 1)



   
        if val_loss < best_val_loss:
            print(f" New best val_loss: {val_loss:.4f} (prev {best_val_loss:.4f})")
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Salva best full model e encoder separatamente
            save_best_model(model, log_dir, epoch + 1)
            print(f" Best model saved at epoch {epoch+1}")

        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")


        if (epoch + 1) % config.get("save_every", 10) == 0:
            # Salva solo il modello completo (per ripristino training)
            save_path = os.path.join(log_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'global_step_val': global_step_val
            }, save_path)
            print(f" Checkpoint (solo full model) salvato in {save_path}")
        
        if epochs_without_improvement >= config["early_stopping_patience"]:
            print("Early stopping triggered.")
            break

    writer.close()

# ==== Run ====

if __name__ == "__main__":
    config = load_config("configs/pretraining.yml")
    train_model(config)


