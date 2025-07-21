
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from tqdm import tqdm
from itertools import combinations
from utils import load_config, BinaryBalancedAccuracy
from CHBMITLoader import make_loader
from model.SupervisedClassifier import BIOTClassifier
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryCohenKappa
)
from torch.utils.tensorboard import SummaryWriter

# Setup DDP
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Leave-one-out splitting utility
def leave_one_out_splits(patients, val_count=2):
    splits = []
    for i, test_patient in enumerate(patients):
        remaining = [p for j, p in enumerate(patients) if j != i]
        val_combinations = list(combinations(remaining, val_count))
        for val_patients in val_combinations:
            train_patients = [p for p in remaining if p not in val_patients]
            splits.append({"train": train_patients, "val": list(val_patients), "test": [test_patient]})
            break
    return splits

# Trainer class
class Trainer:
    def __init__(self, model, optimizer, gpu_id, save_every):
        self.gpu_id = gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[gpu_id])
        self.optimizer = optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.save_every = save_every

    def train_step(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device).float().view(-1, 1)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"[GPU{self.gpu_id}] Training Loss: {avg_loss:.4f}")
        return avg_loss

    def val_step(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).float().view(-1, 1)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                running_loss += loss.item()
        avg_loss = running_loss / len(val_loader)
        print(f"[GPU{self.gpu_id}] Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def test_step(self, test_loader, metrics):
        self.model.eval()
        running_loss = 0.0
        per_file_preds = {}
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).float().view(-1, 1)
                file_ids = batch["file"]
                logits = self.model(x)
                loss = self.criterion(logits, y)
                running_loss += loss.item()
                probs = torch.sigmoid(logits).view(-1)
                y_int = y.long().view(-1)
                
                for m in metrics.values():
                    m.update(probs, y_int)

                for f_id, p, t in zip(file_ids, probs.cpu(), y_int.cpu()):
                    if f_id not in per_file_preds:
                        per_file_preds[f_id] = {"y_true": [], "y_pred": []}
                    per_file_preds[f_id]["y_true"].append(t.item())
                    per_file_preds[f_id]["y_pred"].append(int(p >= 0.5))

        avg_loss = running_loss / len(test_loader)
        print(f"[GPU{self.gpu_id}] Test Loss: {avg_loss:.4f}")

        return avg_loss, per_file_preds
    
    def compute_metrics(self, metrics):
        results = {}

        for metric in metrics.values():
            if torch.distributed.is_initialized():
                metric.sync()  # sincronizza lo stato tra tutti i processi
        for name, metric in metrics.items():
            val = metric.compute()
            # se è tensore, converto con .item(), altrimenti uso direttamente
            if hasattr(val, 'item'):
                val = val.item()
            results[name] = val
        for metric in metrics.values():
            metric.reset()
        return results 

    def supervised(self, config, train_loader, val_loader, test_loader, iteration_idx):
        print(f"Starting supervised training on GPU {self.gpu_id} for iteration {iteration_idx}")
        finetune_mode = config["finetune_mode"]
        writer = SummaryWriter(log_dir=f"{config['log_dir']}/run-{iteration_idx}-{finetune_mode}")

        # Percorsi per checkpoint e best model
        ckpt_path = config["save_dir"].format(iteration_idx=iteration_idx, mode=finetune_mode) + ".ckpt"
        best_model_path = config["save_dir"].format(iteration_idx=iteration_idx, mode=finetune_mode) + "_best_model.pth"

        best_val_loss = float("inf")
        start_epoch = 0
        counter = 0

        # Riprendi da checkpoint se richiesto
        if config["resume"] and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["best_val_loss"]
            counter = checkpoint["counter"]
            print(f"=> Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

        patience = config["early_stopping_patience"]

        for epoch in range(start_epoch, config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")

            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            train_loss = self.train_step(train_loader)
            val_loss = self.val_step(val_loader)

            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)

            # Salvataggio best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best model to {best_model_path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            # Salvataggio checkpoint periodico
            if (epoch + 1) % self.save_every == 0:
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "counter": counter
                }, ckpt_path)
                print(f"=> Checkpoint saved to {ckpt_path}")

        # === Test Step ===
        print(f"\nLoading best model from {best_model_path} for testing")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        test_metrics = {
            "acc": BinaryAccuracy(dist_sync_on_step=False).to(self.device),
            "prauc": BinaryAveragePrecision(dist_sync_on_step=False).to(self.device),
            "auroc": BinaryAUROC(dist_sync_on_step=False).to(self.device),
            "balacc": BinaryBalancedAccuracy(dist_sync_on_step=False).to(self.device),
            "kappa": BinaryCohenKappa(dist_sync_on_step=False).to(self.device)
        }

        test_loss, per_file_preds = self.test_step(test_loader, test_metrics)
        test_results = self.compute_metrics(test_metrics)

        if self.gpu_id == 0:
            print(f"\n=== Split {iteration_idx} Test Results ({finetune_mode}) ===")
            for k, v in test_results.items():
                print(f"{k.upper():7s}: {v:.4f}")
                writer.add_scalar(f"Test/{k}", v)

        print("\n--- Accuracy per file ---")
        for file, results in per_file_preds.items():
            y_true = results["y_true"]
            y_pred = results["y_pred"]
            correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
            acc = correct / len(y_true) if y_true else 0.0
            print(f"{file:35s} | Accuracy: {acc:.4f}")

        if self.gpu_id == 0:
            writer.close()

        return val_loss, test_loss

# Load model and optimizer
def load_train_objs(gpu_id, config, finetune_mode):
    model = BIOTClassifier(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_classes=config["n_classes"]
    )

    optimizer = None
    if finetune_mode in ["frozen_encoder", "full_finetune"]:
        checkpoint = torch.load(config["pretrained_ckpt"], map_location=f"cuda:{gpu_id}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"=> Loaded pretrained weights from {config['pretrained_ckpt']}")

    if finetune_mode == "frozen_encoder":
        for param in model.biot.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif finetune_mode == "full_finetune":
        for param in model.biot.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    elif finetune_mode == "from_scratch":
        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    return model, optimizer

# Main function
def main(rank: int, world_size: int, config: dict):
    ddp_setup(rank, world_size)


    model, optimizer = load_train_objs(rank, config, config["finetune_mode"])
    dataset_path = config["dataset_path"]
    all_patients = sorted([p for p in os.listdir(dataset_path) if not p.startswith(".")])[:6]
    splits = leave_one_out_splits(all_patients, val_count=2)

    for idx, split in enumerate(splits):
        print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")
        train_loader = make_loader(split["train"], dataset_path, config, shuffle=True, is_ddp=True, rank=rank, world_size=world_size)
        val_loader   = make_loader(split["val"], dataset_path, config, shuffle=False, is_ddp=True, rank=rank, world_size=world_size)
        test_loader  = make_loader(split["test"], dataset_path, config, shuffle=False, is_ddp=True, rank=rank, world_size=world_size)

        trainer = Trainer(model, optimizer, rank, config["save_every"])
        trainer.supervised(config, train_loader, val_loader, test_loader, idx + 1)

    destroy_process_group()

if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    world_size = torch.cuda.device_count()

    # Avvia il training parallelo
    mp.spawn(main, args=(world_size, config), nprocs=world_size)

