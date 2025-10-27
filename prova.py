
import os
import argparse
import pickle
import sys
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pyhealth.metrics import binary_metrics_fn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from BIOT_vanilla.biot import BIOTClassifier
from utils import focal_loss, compute_global_stats, load_config
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CHBMITLoader_8s_overlap import make_loader


# ---------------------------------------------------------
#  Funzione per trovare la soglia migliore
# ---------------------------------------------------------
def _pick_threshold(y_true, y_score, metric="bacc", beta=2.0):
    qs = np.linspace(0.0, 1.0, 201)
    cand = np.unique(np.clip(np.quantile(y_score, qs), 1e-8, 1 - 1e-8))
    best_t, best_val = 0.5, -1.0
    best_stats = None

    for t in cand:
        pred = (y_score >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        rec = sens

        if metric == "bacc":
            val = 0.5 * (sens + spec)
        elif metric == "fbeta":
            # formula generale di F_beta
            val = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-12)
        else:
            val = 0.5 * (sens + spec)

        if val > best_val:
            best_val = val
            best_t = float(t)
            best_stats = (sens, spec, prec, rec)

    return best_t, best_val, best_stats


# ---------------------------------------------------------
#  Lightning Module
# ---------------------------------------------------------
class LitModel_finetune(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.threshold = config.get("threshold", 0.5)
        self.config = config
        self.alpha_focal = config["focal_alpha"]
        self.gamma_focal = config["focal_gamma"]
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_results = {"preds": [], "targets": []}
        self.test_results = {"preds": [], "targets": []}

    def training_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        y = y.float().unsqueeze(1)
        logits = self.model(X)
        #loss = self.criterion(logits, y)
        loss = focal_loss(logits, y, alpha=self.alpha_focal, gamma=self.gamma_focal)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        y = y.float().unsqueeze(1)
        with torch.no_grad():
            logits = self.model(X)
            #loss = self.criterion(logits, y)
            loss = focal_loss(logits, y, alpha=self.alpha_focal, gamma=self.gamma_focal)
            step_result = torch.sigmoid(logits).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.val_results["preds"].append(step_result)
        self.val_results["targets"].append(step_gt)
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        result = np.concatenate(self.val_results["preds"])
        gt = np.concatenate(self.val_results["targets"])

        if sum(gt) * (len(gt) - sum(gt)) != 0:
            #self.threshold, best_score, (sens, spec, prec, rec) = _pick_threshold(
            #    gt, result, metric="bacc", beta=2.0
            #)
            self.threshold = self.config.get("threshold", 0.5)
            print(f"  Nuova soglia ottimale: {self.threshold:.4f}")

            preds_bin = (result >= self.threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(gt, preds_bin, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-12)
            specificity = tn / (tn + fp + 1e-12)
            precision = tp / (tp + fp + 1e-12)
            recall = sensitivity

            results = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )

            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
            results["precision"] = precision
            results["recall"] = recall
        else:
            results = {k: 0.0 for k in [
                "accuracy", "balanced_accuracy", "pr_auc", "roc_auc",
                "sensitivity", "specificity", "precision", "recall"
            ]}

        self.log("val_acc", results["accuracy"], sync_dist=True)
        self.log("val_bacc", results["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", results["pr_auc"], sync_dist=True)
        self.log("val_auroc", results["roc_auc"], sync_dist=True)
        self.log("val_sensitivity", results["sensitivity"], sync_dist=True)
        self.log("val_specificity", results["specificity"], sync_dist=True)
        self.log("val_precision", results["precision"], sync_dist=True)
        self.log("val_recall", results["recall"], sync_dist=True)
        self.log("val_prevalence", gt.mean(), sync_dist=True)

        print({k: float(v) for k, v in results.items()})
        self.val_results = {"preds": [], "targets": []}
        return results

    def test_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.test_results["preds"].append(step_result)
        self.test_results["targets"].append(step_gt)

    def on_test_epoch_end(self):
        result = np.concatenate(self.test_results["preds"])
        gt = np.concatenate(self.test_results["targets"])

        if sum(gt) * (len(gt) - sum(gt)) != 0:
            self.threshold = self.config.get("threshold", 0.5)
            results = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
            preds_bin = (result >= self.threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(gt, preds_bin, labels=[0, 1]).ravel()
            sensitivity = tp / (tp + fn + 1e-12)
            specificity = tn / (tn + fp + 1e-12)
            precision = tp / (tp + fp + 1e-12)
            recall = sensitivity

            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
            results["precision"] = precision
            results["recall"] = recall
        else:
            results = {k: 0.0 for k in [
                "accuracy", "balanced_accuracy", "pr_auc", "roc_auc",
                "sensitivity", "specificity", "precision", "recall"
            ]}

        self.log("test_acc", results["accuracy"], sync_dist=True)
        self.log("test_bacc", results["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", results["pr_auc"], sync_dist=True)
        self.log("test_auroc", results["roc_auc"], sync_dist=True)
        self.log("test_sensitivity", results["sensitivity"], sync_dist=True)
        self.log("test_specificity", results["specificity"], sync_dist=True)
        self.log("test_precision", results["precision"], sync_dist=True)
        self.log("test_recall", results["recall"], sync_dist=True)
        self.log("test_prevalence", gt.mean(), sync_dist=True)

        print({k: float(v) for k, v in results.items()})
        self.test_results = {"preds": [], "targets": []}
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=float(self.config["weight_decay"]),
        )
        return [optimizer]

    # Salva / carica soglia nel checkpoint
    def on_save_checkpoint(self, checkpoint):
        checkpoint["best_threshold"] = float(getattr(self, "threshold", 0.5))

    def on_load_checkpoint(self, checkpoint):
        if "best_threshold" in checkpoint:
            self.threshold = float(checkpoint["best_threshold"])


# ---------------------------------------------------------
#  Data Preparation
# ---------------------------------------------------------
def predefined_split():
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 17)]
    val_patients = [f"chb{str(i).zfill(2)}" for i in range(17, 21)]
    test_patients = [f"chb{str(i).zfill(2)}" for i in range(21, 24)]
    return {"train": train_patients, "val": val_patients, "test": test_patients}


def prepare_CHB_MIT_dataloader(config):
    dataset_path = config["dataset_path"]
    gt_path = config["gt_path"]

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    split = predefined_split()

    train_loader = make_loader(
        split["train"], dataset_path, gt_path, config,
        shuffle=True, balanced=True, neg_to_pos_ratio=5
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
#  Training & Evaluation
# ---------------------------------------------------------
def supervised(config):
    train_loader, val_loader, test_loader = prepare_CHB_MIT_dataloader(config)

    model = BIOTClassifier(
        n_channels=config["n_channels"],
        n_fft=200,
        hop_length=100,
    )

    lightning_model = LitModel_finetune(config, model)

    version = f"CHB-MIT-{config['finetune_mode']}"
    logger = TensorBoardLogger(save_dir="./", version=version, name="log")

    early_stop_callback = EarlyStopping(
        monitor="val_bacc",
        patience=config["early_stopping_patience"],
        verbose=False,
        mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_bacc",
        mode="max",
        save_top_k=1,
        filename="best-model"
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy="auto",
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=config["epochs"],
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # Addestramento + early stopping sulla validation reale
    trainer.fit(lightning_model, train_loader, val_loader)

    # Ricarica i migliori pesi
    best_path = checkpoint_callback.best_model_path
    best_model = LitModel_finetune.load_from_checkpoint(
        best_path, config=config, model=BIOTClassifier(
            n_channels=config["n_channels"],
            n_fft=200,
            hop_length=100,
        )
    )

    # Trova soglia ottimale su validation bilanciata
    print("\n===> Calcolo soglia ottimale su validation bilanciata...")
    trainer.validate(model=best_model, dataloaders=val_loader)

    # Valuta su test reale
    print("\n===> Test finale su distribuzione reale...")
    test_results = trainer.test(model=best_model, dataloaders=test_loader)[0]
    print("Test results:", test_results)

    # Metriche su validation reale (solo report, no threshold tuning)
    val_metrics = trainer.validate(model=best_model, dataloaders=val_loader)[0]
    return val_metrics

# ---------------------------------------------------------
#  Optuna
# ---------------------------------------------------------
import optuna
import pandas as pd

def objective(trial):
    config = load_config("configs/finetuning.yml")
    config["lr"] = trial.suggest_loguniform("lr", 1e-6, 1e-4)
    config["focal_alpha"] = trial.suggest_uniform("focal_alpha", 0.2, 0.9)
    config["focal_gamma"] = trial.suggest_uniform("focal_gamma", 1.0, 5.0)
    config["weight_decay"] = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
    config["threshold"] = trial.suggest_float("threshold", 0.2, 0.7)
    config["epochs"] = 100

    results = supervised(config)
    return results["val_bacc"]

if __name__ == "__main__":
    storage_name = "sqlite:///optuna_finetuning_val_bacc.db"
    study_name = "finetuning_tuning_val_bacc"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=storage_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=15)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    for k, v in trial.params.items():
        print(f"  {k}: {v}")

    df = study.trials_dataframe()
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{study_name}_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n Risultati salvati in {output_path}")


# if __name__ == "__main__":
#     config = load_config("configs/finetuning.yml")
#     supervised(config)