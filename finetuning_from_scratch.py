import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
from tqdm import tqdm
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC,
    BinarySpecificity, BinaryRecall
)

from utils import load_config
from CHBMITLoader_8s_overlap import make_loader
from model.SupervisedClassifier import BIOTClassifier
from sklearn.model_selection import KFold


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# Trainer minimale
# ---------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        criterion_name = config.get("criterion_name", "focal").lower()
        if criterion_name == "focal":
            alpha = float(config.get("focal_alpha", 0.5))
            gamma = float(config.get("focal_gamma", 2.0))
            self.criterion = lambda logits, y: sigmoid_focal_loss(
                inputs=logits.view(-1),
                targets=y.view(-1),
                alpha=alpha,
                gamma=gamma,
                reduction="mean"
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Metriche
        self.metrics = {
            "acc": BinaryAccuracy().to(self.device),
            "prauc": BinaryAveragePrecision().to(self.device),
            "auroc": BinaryAUROC().to(self.device),
            "spec": BinarySpecificity().to(self.device),
            "sens": BinaryRecall().to(self.device),
        }

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0
        for batch in tqdm(loader, desc="Train", leave=False):
            x, y = batch["x"].to(self.device), batch["y"].to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in loader:
                x, y = batch["x"].to(self.device), batch["y"].to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += loss.item()

                probs = torch.sigmoid(logits).view(-1)
                y_int = y.view(-1).long()
                for m in self.metrics.values():
                    m.update(probs, y_int)

        results = {k: v.compute().item() for k, v in self.metrics.items()}
        for v in self.metrics.values():
            v.reset()

        results["balacc"] = 0.5 * (results["sens"] + results["spec"])
        return total_loss / len(loader), results

    def fit(self, train_loader, val_loader, test_loader, config):
        epochs = int(config.get("epochs", 50))
        patience = int(config.get("early_stopping_patience", 10))
        best_val = -1.0
        counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_res = self.evaluate(val_loader)
            val_metric = val_res["prauc"]

            if self.scheduler is not None:
                self.scheduler.step(val_metric)

            improved = val_metric > best_val
            if improved:
                best_val = val_metric
                best_state = self.model.state_dict()
                counter = 0
            else:
                counter += 1

            print(f"[Epoch {epoch+1:03d}] "
                  f"TrainLoss={train_loss:.4f} "
                  f"Val_prAUC={val_metric:.4f} "
                  f"BalAcc={val_res['balacc']:.3f}")

            if counter >= patience:
                print("Early stopping.")
                break

        # Test best pesi
        if best_state is not None:
            self.model.load_state_dict(best_state)
        _, test_res = self.evaluate(test_loader)
        return best_val, test_res


# ---------------------------------------------------------
# K-Fold paziente-wise
# ---------------------------------------------------------
def patientwise_splits(num_patients=23, test_patient_idx=23, n_splits=5, seed=42):
    """
    Genera 5 fold cross-validation paziente-wise.
    """
    assert 1 <= test_patient_idx <= num_patients
    all_patients = [f"chb{str(i).zfill(2)}" for i in range(1, num_patients + 1)]
    always_train = ["chb21", "chb22"]

    test_patient = f"chb{str(test_patient_idx).zfill(2)}"
    cv_candidates = [p for p in all_patients if p not in [test_patient, *always_train]]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(cv_candidates):
        train_pat = [cv_candidates[i] for i in train_idx] + always_train
        val_pat = [cv_candidates[i] for i in val_idx]
        folds.append({"train": train_pat, "val": val_pat, "test": [test_patient]})
    return folds


# ---------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------
def make_data_loaders(config, split):
    dataset_path = config["dataset_path"]
    gt_path = config["gt_path"]

    train_loader = make_loader(split["train"], dataset_path, gt_path, config,
                               shuffle=True, balanced=True, neg_to_pos_ratio=5)
    val_loader = make_loader(split["val"], dataset_path, gt_path, config,
                             shuffle=False, balanced=False)
    test_loader = make_loader(split["test"], dataset_path, gt_path, config,
                              shuffle=False, balanced=False)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    config = load_config("configs/finetuning.yml")
    set_seed(42)

    folds = patientwise_splits(num_patients=23, test_patient_idx=23, n_splits=5, seed=42)
    for i, split in enumerate(folds, 1):
        print(f"\n===== Fold {i}/5 =====")
        print("Train:", split["train"])
        print("Val:  ", split["val"])
        print("Test: ", split["test"])

        train_loader, val_loader, test_loader = make_data_loaders(config, split)

        model = BIOTClassifier(
            n_channels=int(config.get("n_channels", 16)),
            n_fft=int(config.get("n_fft", 200)),
            hop_length=int(config.get("hop_length", 100)),
            num_classes=1
        )

        optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config.get("lr", 5e-5)),
            weight_decay=float(config.get("weight_decay", 1e-6))
        )

        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        trainer = Trainer(model, optimizer, scheduler, config)
        best_val, test_res = trainer.fit(train_loader, val_loader, test_loader, config)

        print(f"Fold {i} — Best val prAUC: {best_val:.4f}")
        print("Test results:")
        for k, v in test_res.items():
            print(f"  {k}: {v:.4f}")


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    cv_folds = patientwise_splits(num_patients=23, test_patient_idx=None, n_splits=5, seed=42)
    for i, fold in enumerate(cv_folds):
        print(f"Fold {i+1}:")
        print(" Train:", fold["train"])
        print(" Val:  ", fold["val"])
        print(" Test: ", fold["test"])
        print()