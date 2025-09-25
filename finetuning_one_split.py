


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from itertools import combinations
from utils import load_config, BinaryBalancedAccuracy, compute_global_stats, focal_loss
from CHBMITLoader import make_loader
from model.SupervisedClassifier import BIOTClassifier
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryCohenKappa
)
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import random
import numpy as np




# Leave-one-out splitting utility (rimane uguale)
from itertools import combinations
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


# def compute_pos_weight(train_loader, device):
#     total_pos, total_neg = 0, 0
#     for batch in train_loader:
#         y = batch["y"].view(-1)
#         total_pos += (y == 1).sum().item()
#         total_neg += (y == 0).sum().item()
#     pos_weight = torch.tensor([total_neg / max(total_pos, 1)], device=device)
#     print(f"Computed pos_weight: {pos_weight.item():.4f} (neg={total_neg}, pos={total_pos})")
#     return pos_weight

# Trainer con DataParallel
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion_name,save_every, pos_weight= None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model = nn.DataParallel(self.model) 
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pos_weight = pos_weight
        self.save_every = save_every

        self.val_metrics = {
        "acc": BinaryAccuracy().to(self.device),
        "prauc": BinaryAveragePrecision().to(self.device),
        "auroc": BinaryAUROC().to(self.device),
        "balacc": BinaryBalancedAccuracy().to(self.device),
        "kappa": BinaryCohenKappa().to(self.device),
        }

        if criterion_name == "focal":
            self.criterion = focal_loss
         
        if criterion_name == "bce":
            assert self.pos_weight is not None, "pos_weight must be provided for BCE"
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        
        else:
            raise ValueError(f"Unknown criterion: {criterion_name}")

    def train_step(self, train_loader):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(train_loader, desc="Training", dynamic_ncols=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device).float().view(-1, 1)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        print(f"Training Loss: {avg_loss:.4f}")
        return avg_loss

    def val_step(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", dynamic_ncols=False):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).float().view(-1, 1)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                running_loss += loss.item()

                probs = torch.sigmoid(logits).view(-1)
                y_int = y.long().view(-1)
                for m in self.val_metrics.values():
                    m.update(probs, y_int)

        avg_loss = running_loss / len(val_loader)
        val_results = self.compute_metrics(self.val_metrics)

        print(f"Validation Loss: {avg_loss:.8f}")
        print(f"Validation Metrics: {val_results}")
        
        return avg_loss, val_results

    def test_step(self, test_loader, metrics):
        self.model.eval()
        running_loss = 0.0
       
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing", dynamic_ncols=False):
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device).float().view(-1, 1) 
                logits = self.model(x)
                loss = self.criterion(logits, y)
                running_loss += loss.item()
                probs = torch.sigmoid(logits).view(-1)
                y_int = y.long().view(-1)

                for m in metrics.values():
                    m.update(probs, y_int)

              

        avg_loss = running_loss / len(test_loader)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def compute_metrics(self, metrics):
        results = {}
        for name, metric in metrics.items():
            val = metric.compute()
            if hasattr(val, "item"):
                val = val.item()
            results[name] = val
        for metric in metrics.values():
            metric.reset()
        return results

    def supervised(self, config, train_loader, val_loader, test_loader, iteration_idx):
        print(f"Starting supervised training for iteration {iteration_idx}")
        finetune_mode = config["finetune_mode"]
        writer = SummaryWriter(log_dir=f"{config['log_dir']}/run-{iteration_idx}-{finetune_mode}")

        run_dir = os.path.join(config["save_dir"], f"run-{iteration_idx}-{finetune_mode}")
        os.makedirs(run_dir, exist_ok=True)
        ckpt_path = os.path.join(run_dir, "checkpoint.ckpt")
        best_model_path = os.path.join(run_dir, "best_model.pth")

       
        best_val_metric = -float("inf")  # inizializza a -inf
        start_epoch = 0
        counter = 0

        if config["resume"] and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_metric = checkpoint["best_val_metric"]
            counter = checkpoint["counter"]
            print(f"=> Resumed from epoch {start_epoch}, best_val_metric={best_val_metric:.4f}")

        patience = config["early_stopping_patience"]

        

        for epoch in range(start_epoch, config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")
            train_loss = self.train_step(train_loader)
            val_loss, val_results = self.val_step(val_loader)

            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)

            for k, v in val_results.items():
                writer.add_scalar(f"Val/{k}", v, epoch + 1)

            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step(val_results["prauc"])

            current_prauc = val_results["prauc"]

            if current_prauc > best_val_metric:
                best_val_metric = current_prauc
                counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Saved best model (PR-AUC={current_prauc:.4f}) to {best_model_path}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no PR-AUC improvement)")
                    break

            # if val_loss < best_val_loss:
            #     best_val_loss = val_loss
            #     counter = 0
            #     torch.save(self.model.state_dict(), best_model_path)
            #     print(f"Saved best model to {best_model_path}")
            # else:
            #     counter += 1
            #     #if counter >= patience:
            #     #    print(f"Early stopping at epoch {epoch + 1}")
            #     #    break

            if (epoch + 1) % self.save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict() if self.scheduler else None,
                    "best_val_metric": best_val_metric,
                    "counter": counter
                }, ckpt_path)
                print(f"=> Checkpoint saved to {ckpt_path}")


        print(f"\nLoading best model from {best_model_path} for testing")
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        test_metrics = {
            "acc": BinaryAccuracy().to(self.device),
            "prauc": BinaryAveragePrecision().to(self.device),
            "auroc": BinaryAUROC().to(self.device),
            "balacc": BinaryBalancedAccuracy().to(self.device),
            "kappa": BinaryCohenKappa().to(self.device),
        }

        test_loss = self.test_step(test_loader, test_metrics)
        test_results = self.compute_metrics(test_metrics)

        print(f"\n=== Split {iteration_idx} Test Results ({finetune_mode}) ===")
        for k, v in test_results.items():
            print(f"{k.upper():7s}: {v:.4f}")
            writer.add_scalar(f"Test/{k}", v)

    
        writer.close()
        results_dict = {
        "iteration": iteration_idx,
        "finetune_mode": finetune_mode,
        "test_loss": test_loss,
        "metrics": test_results
        }

        json_path = os.path.join(run_dir, f"results_split{iteration_idx}.json")
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=4)

        print(f"=> Saved test results to {json_path}")


        return val_loss, test_loss


# Load model and optimizer
def load_train_objs(gpu_id, config, finetune_mode, resume=False):
    model = BIOTClassifier(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_classes=config["n_classes"]
    )

    optimizer = None
    if not resume and finetune_mode in ["frozen_encoder", "full_finetune"]:
        checkpoint = torch.load(config["pretrained_ckpt"], map_location=f"cuda:{gpu_id}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(f"=> Loaded pretrained weights from {config['pretrained_ckpt']}")

    if finetune_mode == "frozen_encoder":
        for param in model.biot.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    elif finetune_mode == "full_finetune":
        for param in model.parameters():
            if param.dtype.is_floating_point or param.is_complex():
                param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    elif finetune_mode == "from_scratch":
        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")
    


    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=5
    )
    return model, optimizer, scheduler


def predefined_split():
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    return {"train": train_patients, "val": val_patients, "test": test_patients}


# Main function
def main(config: dict):
    model, optimizer, scheduler = load_train_objs(0, config, config["finetune_mode"], config["resume"])
    dataset_path = config["dataset_path"]
    gt_path = "../../Datasets/chb_mit/GT"


    #all_patients = sorted([p for p in os.listdir(dataset_path) if not p.startswith(".")])[:6]
    #splits = leave_one_out_splits(all_patients, val_count=2)

    split = predefined_split()

    # for i, split in enumerate(splits):
    #     print(f"Split {i+1}:")
    #     print(f"  Train: {split['train']}")
    #     print(f"  Val:   {split['val']}")
    #     print(f"  Test:  {split['test']}")
    

    #     for idx, split in enumerate(splits):
    #         print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


    #train_mean, train_std = compute_global_stats(split["train"], dataset_path)
    #mean_t = torch.tensor(train_mean, dtype=torch.float32).view(18, 1)
    #std_t = torch.tensor(train_std, dtype=torch.float32).view(18, 1)
    #print("train_mean:", train_mean)
    #print("train_std:", train_std)
    
    train_loader = make_loader(split["train"], dataset_path, gt_path, config,  balanced=True, shuffle=True)
    val_loader   = make_loader(split["val"], dataset_path, gt_path, config, shuffle=False)
    test_loader  = make_loader(split["test"], dataset_path, gt_path, config, shuffle=False)
    
    # Calcolo pos_weight
    #pos_weight = compute_pos_weight(train_loader, device="cuda")
    
    trainer = Trainer(model, optimizer, scheduler, criterion_name= config["criterion_name"], save_every=config["save_every"])
    trainer.supervised(config, train_loader, val_loader, test_loader, 1)


if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    main(config)