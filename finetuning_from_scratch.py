import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import random
import json
import optuna
from tqdm import tqdm

from utils import load_config
from CHBMITLoader_8s_overlap import make_loader
from model.SupervisedClassifier import BIOTClassifier
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC,
    BinarySpecificity, BinaryRecall
)
from torchmetrics.functional.classification import binary_focal_loss


# ---------------------------------------------------------
# Utility
# ---------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predefined_split():
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients   = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients  = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    return {"train": train_patients, "val": val_patients, "test": test_patients}


# ---------------------------------------------------------
# Trainer minimale
# ---------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, scheduler, config, criterion_name):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Loss
        if criterion_name == "focal":
            self.alpha = float(config.get("focal_alpha", 0.5))
            self.gamma = float(config.get("focal_gamma", 2.0))
            self.criterion = lambda logits, y: binary_focal_loss(
                preds=logits.view(-1),
                target=y.view(-1).to(torch.long),
                alpha=self.alpha, gamma=self.gamma,
                reduction="mean", logits=True
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
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

    def fit(self, train_loader, val_loader, test_loader, config, trial=None):
        epochs = int(config.get("epochs", 50))
        patience = int(config.get("early_stopping_patience", 10))
        best_val = -1.0
        counter = 0

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

            print(f"[Epoch {epoch+1}] train_loss={train_loss:.4f} "
                  f"val_pr_auc={val_metric:.4f} balacc={val_res['balacc']:.3f}")

            if trial is not None:
                trial.report(val_metric, step=epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if counter >= patience:
                print("Early stopping.")
                break

        # Test best
        self.model.load_state_dict(best_state)
        _, test_res = self.evaluate(test_loader)
        return best_val, test_res


# ---------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------
def objective(trial, base_config):
    set_seed(42 + trial.number)

    # hyperparams to tune
    lr = trial.suggest_float("lr", 1e-5, 3e-4, log=True)
    wd = trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True)
    crit = trial.suggest_categorical("criterion_name", ["bce", "focal"])

    config = dict(base_config)
    config.update({"lr": lr, "weight_decay": wd, "criterion_name": crit})

    model = BIOTClassifier(
        emb_size=config["emb_size"],
        heads=config["heads"],
        depth=config["depth"],
        n_classes=config["n_classes"],
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)

    trainer = Trainer(model, optimizer, scheduler, config, crit)

    split = predefined_split()
    train_loader = make_loader(split["train"], config["dataset_path"], "../../Datasets/chb_mit/GT",
                               config, shuffle=True, balanced=True, neg_to_pos_ratio=5)
    val_loader   = make_loader(split["val"], config["dataset_path"], "../../Datasets/chb_mit/GT",
                               config, shuffle=False)
    test_loader  = make_loader(split["test"], config["dataset_path"], "../../Datasets/chb_mit/GT",
                               config, shuffle=False)

    best_val, test_results = trainer.fit(train_loader, val_loader, test_loader, config, trial=trial)
    print("→ Final Test:", test_results)

    return best_val


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    cfg = load_config("configs/finetuning.yml")
    set_seed(int(cfg.get("seed", 42)))

    study = optuna.create_study(
        direction="maximize",
        study_name=cfg.get("study_name", "BIOT_from_scratch"),
        storage="sqlite:///optuna_biot_from_scratch.db",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )

    n_trials = int(cfg.get("n_trials", 20))
    study.optimize(lambda t: objective(t, cfg), n_trials=n_trials, gc_after_trial=True)

    print("Best trial:")
    print(" Value:", study.best_value)
    print(" Params:", study.best_trial.params)

import itertools
import numpy as np
from sklearn.model_selection import KFold

def patientwise_splits(num_patients=23, test_patient_idx=None, n_splits=5, seed=42):
    """
    Genera 5-fold cross-validation paziente-wise per CHB-MIT.
    
    Args:
        num_patients: totale pazienti (default 23)
        test_patient_idx: paziente scelto come test (1-based)
        n_splits: numero fold interni di validation
        seed: random state
        
    Returns:
        folds: lista di dict { "train": [...], "val": [...], "test": [...] }
    """
    assert 1 <= test_patient_idx <= num_patients, "Indice test non valido"
    
    all_patients = [f"chb{str(i).zfill(2)}" for i in range(1, num_patients + 1)]

    # Pazienti 21 e 22 sempre in TRAIN (esclusi dalla CV)
    always_train = ["chb21", "chb22"]

    # Definisci test e candidati per k-fold
    test_patient = f"chb{str(test_patient_idx).zfill(2)}"
    cv_candidates = [p for p in all_patients if p not in [test_patient, *always_train]]

    # Crea 5 fold patient-wise
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_folds = []
    for train_idx, val_idx in kf.split(cv_candidates):
        train_pat = [cv_candidates[i] for i in train_idx] + always_train
        val_pat   = [cv_candidates[i] for i in val_idx]
        fold = {"train": train_pat, "val": val_pat, "test": [test_patient]}
        cv_folds.append(fold)

    return cv_folds

if __name__ == "__main__":
    cv_folds = patientwise_splits(num_patients=23, test_patient_idx=None, n_splits=5, seed=42)
    for i, fold in enumerate(cv_folds):
        print(f"Fold {i+1}:")
        print(" Train:", fold["train"])
        print(" Val:  ", fold["val"])
        print(" Test: ", fold["test"])
        print()