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
from sklearn.model_selection import KFold

from utils import load_config
from CHBMITLoader_8s_overlap import make_loader

from BIOT_vanilla.biot import BIOTClassifier

# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_gt_path(cfg):
    return cfg.get("gt_path") or cfg.get("GT_path") or "../../Datasets/chb_mit/GT"


# ---------------------------------------------------------
# Trainer minimale
# ---------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss
        criterion_name = str(config.get("criterion_name", "focal")).lower()
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

        # Metriche (threshold-free dove possibile)
        self.metrics = {
            "acc":  BinaryAccuracy().to(self.device),
            "prauc": BinaryAveragePrecision().to(self.device),
            "auroc": BinaryAUROC().to(self.device),
            "spec": BinarySpecificity().to(self.device),
            "sens": BinaryRecall().to(self.device),
        }

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc="Train", leave=False):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
        return total_loss / max(1, len(loader))

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)
                total_loss += float(loss.item())

                probs = torch.sigmoid(logits).view(-1)
                y_int = y.view(-1).long()
                for m in self.metrics.values():
                    m.update(probs, y_int)

        results = {k: v.compute().item() for k, v in self.metrics.items()}
        for v in self.metrics.values():
            v.reset()

        # Balanced Accuracy = (sens + spec) / 2
        results["balacc"] = 0.5 * (results["sens"] + results["spec"])
        return total_loss / max(1, len(loader)), results

    def fit(self, train_loader, val_loader, test_loader, config):
        epochs = int(config.get("epochs", 50))
        patience = int(config.get("early_stopping_patience", 10))

        best_val = -1.0
        best_state = None
        counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_res = self.evaluate(val_loader)
            val_metric = float(val_res["prauc"])

            if self.scheduler is not None:
                self.scheduler.step(val_metric)

            improved = val_metric > best_val
            if improved:
                best_val = val_metric
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                counter = 0
            else:
                counter += 1

            print(f"[Epoch {epoch:03d}] "
                  f"TrainLoss={train_loss:.4f}  ValLoss={val_loss:.4f}  "
                  f"Val_prAUC={val_metric:.4f}  Val_balAcc={val_res['balacc']:.4f}")

            if counter >= patience:
                print("Early stopping.")
                break

        # Test con i pesi migliori
        if best_state is not None:
            self.model.load_state_dict(best_state, strict=True)
        _, test_res = self.evaluate(test_loader)
        return best_val, test_res


# ---------------------------------------------------------
# Splits: 5-fold patient-wise con test fissato e chb21/chb22 sempre in train
# ---------------------------------------------------------
def patientwise_splits(num_patients=23, test_patient_idx=23, n_splits=5, seed=42):
    assert 1 <= test_patient_idx <= num_patients, "Indice test non valido"
    all_patients = [f"chb{str(i).zfill(2)}" for i in range(1, num_patients + 1)]

    always_train = ["chb21", "chb22"]
    test_patient = f"chb{str(test_patient_idx).zfill(2)}"

    # Candidati per la CV (escludi test e always_train)
    cv_candidates = [p for p in all_patients if p not in [test_patient, *always_train]]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, val_idx in kf.split(cv_candidates):
        train_pat = [cv_candidates[i] for i in train_idx] + always_train
        val_pat   = [cv_candidates[i] for i in val_idx]
        folds.append({"train": train_pat, "val": val_pat, "test": [test_patient]})
    return folds


# ---------------------------------------------------------
# DataLoader builder
# ---------------------------------------------------------
def make_data_loaders(config, split):
    dataset_path = config["dataset_path"]
    gt_path = get_gt_path(config)

    train_loader = make_loader(
        split["train"], dataset_path, gt_path, config,
        shuffle=True, balanced=True, neg_to_pos_ratio=int(config.get("neg_to_pos_ratio", 5))
    )
    val_loader = make_loader(
        split["val"], dataset_path, gt_path, config,
        shuffle=False, balanced=False
    )
    test_loader = make_loader(
        split["test"], dataset_path, gt_path, config,
        shuffle=False, balanced=False
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    config = load_config("configs/finetuning.yml")
    set_seed(int(config.get("seed", 42)))

    # Parametri modello
    n_channels  = int(config.get("n_channels", 16))
    n_fft       = int(config.get("n_fft", 200))
    hop_length  = int(config.get("hop_length", 100))
    num_classes = 1

    # Parametri trainer
    lr           = float(config.get("lr", 5e-5))
    weight_decay = float(config.get("weight_decay", 1e-6))

    test_patient_idx = int(config.get("test_patient_idx", 23))
    n_folds = int(config.get("n_folds", 5))

    folds = patientwise_splits(
        num_patients=23,
        test_patient_idx=test_patient_idx,
        n_splits=n_folds,
        seed=int(config.get("cv_seed", 42))
    )

    all_val_pr_auc = []
    all_test_metrics = []

    print(f"\n===> TEST PATIENT: chb{str(test_patient_idx).zfill(2)} | Folds: {n_folds}\n")

    for i, split in enumerate(folds, 1):
        print(f"===== Fold {i}/{n_folds} =====")
        print("Train:", split["train"])
        print("Val:  ", split["val"])
        print("Test: ", split["test"])

        train_loader, val_loader, test_loader = make_data_loaders(config, split)

        model = BIOTClassifier(
            n_channels=n_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            num_classes=num_classes
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        trainer = Trainer(model, optimizer, scheduler, config)
        best_val, test_res = trainer.fit(train_loader, val_loader, test_loader, config)

        all_val_pr_auc.append(best_val)
        all_test_metrics.append(test_res)

        print(f"Fold {i} — Best Val PR-AUC: {best_val:.4f}")
        print("Fold Test metrics:")
        for k, v in test_res.items():
            print(f"  {k:7s}: {v:.4f}")
        print()

    # Media finale sui fold
    print("===== AVERAGE OVER FOLDS =====")
    print(f"Mean Val PR-AUC: {np.mean(all_val_pr_auc):.4f} ± {np.std(all_val_pr_auc):.4f}")

    keys = list(all_test_metrics[0].keys())
    mean_metrics = {k: float(np.mean([d[k] for d in all_test_metrics])) for k in keys}
    std_metrics  = {k: float(np.std([d[k] for d in all_test_metrics]))  for k in keys}

    for k in keys:
        print(f"{k:7s}: {mean_metrics[k]:.4f} ± {std_metrics[k]:.4f}")


if __name__ == "__main__":
    main()