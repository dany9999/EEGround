import os
import argparse
import pickle
import sys
import random
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import re
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pyhealth.metrics import binary_metrics_fn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix

from BIOT_vanilla.biot import BIOTClassifier
from utils import focal_loss, load_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CHBMITLoader_4s import make_loader, compute_global_channel_stats


# ==========================================================
# Utility: caricamento pesi pretrained
# ==========================================================
def load_pretrained_encoder_into_biot(model, ckpt_path, device="cpu"):
    print(f"\n Caricamento pesi pretrainati da: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    new_sd = {}
    for k, v in sd.items():
        k_clean = k[len("module."):] if k.startswith("module.") else k
        if k_clean.startswith("encoder."):
            k_clean = re.sub(r"^encoder\.", "biot.", k_clean)
        elif not k_clean.startswith("biot."):
            k_clean = "biot." + k_clean
        new_sd[k_clean] = v

    model_dict = model.state_dict()
    compatible = {k: v for k, v in new_sd.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    print(f"  → Caricati {len(compatible)} layer compatibili.")
    return model


# ==========================================================
# LightningModule per fine-tuning
# ==========================================================
class LitModel_finetune(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.config = config
        self.threshold = config["threshold"]
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_results = {"preds": [], "targets": []}
        self.test_results = {"preds": [], "targets": []}

    def training_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"].float().unsqueeze(1)
        logits = self.model(X)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"].float().unsqueeze(1)
        with torch.no_grad():
            logits = self.model(X)
            loss = self.criterion(logits, y)
            preds = torch.sigmoid(logits).cpu().numpy()
        self.val_results["preds"].append(preds)
        self.val_results["targets"].append(y.cpu().numpy())
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        result = np.concatenate(self.val_results["preds"]).ravel()
        gt = np.concatenate(self.val_results["targets"]).ravel()
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            #self.threshold = np.sort(result)[-int(np.sum(gt))]
            results = binary_metrics_fn(gt, result,
                                        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                                        threshold=self.threshold)
            preds_bin = (result >= self.threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(gt, preds_bin).ravel()
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
        else:
            results = {m: 0.0 for m in ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "sensitivity", "specificity"]}

        for k, v in results.items():
            self.log(f"val_{k}", float(v), sync_dist=True)
        self.val_results = {"preds": [], "targets": []}
        return results

    def test_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"].float().unsqueeze(1)
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.sigmoid(logits).cpu().numpy()
        self.test_results["preds"].append(preds)
        self.test_results["targets"].append(y.cpu().numpy())

    def on_test_epoch_end(self):
        result = np.concatenate(self.test_results["preds"]).ravel()
        gt = np.concatenate(self.test_results["targets"]).ravel()
        if sum(gt) * (len(gt) - sum(gt)) != 0:
            results = binary_metrics_fn(gt, result,
                                        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                                        threshold=self.threshold)
            preds_bin = (result >= self.threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(gt, preds_bin).ravel()
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
        else:
            results = {m: 0.0 for m in ["accuracy", "balanced_accuracy", "pr_auc", "roc_auc", "sensitivity", "specificity"]}
        self.test_results = {"preds": [], "targets": []}
        return results

    def configure_optimizers(self):
        if self.config["finetune_mode"] == "full_finetune":
            encoder_params = [p for n, p in self.named_parameters() if "biot" in n]
            head_params = [p for n, p in self.named_parameters() if "biot" not in n]
            optimizer = torch.optim.Adam([
                {"params": encoder_params, "lr": self.config["encoder_lr"]},
                {"params": head_params, "lr": self.config["head_lr"]},
            ], weight_decay=float(self.config["weight_decay"]))
        else:
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config["lr"],
                                         weight_decay=float(self.config["weight_decay"]))
        return optimizer


# ==========================================================
# DataLoader per k-fold
# ==========================================================
def make_data_loaders_cv(config, train_patients, val_patients, test_patients):
    dataset_path = config["dataset_path_4s"]
    gt_path = "../../Datasets/chb_mit/GT"

    # Calcolo mu e sigma SOLO sul train del fold
    tmp_loader = make_loader(train_patients, dataset_path, gt_path, config,
                             shuffle=False, balanced=False)
    mu, sigma = compute_global_channel_stats(tmp_loader, n_channels=config["n_channels"])

    train_loader = make_loader(train_patients, dataset_path, gt_path, config,
                               shuffle=True, balanced=True, neg_to_pos_ratio=5, mu=mu, sigma=sigma)
    val_loader = make_loader(val_patients, dataset_path, gt_path, config,
                             shuffle=False, balanced=False, mu=mu, sigma=sigma)
    test_loader = make_loader(test_patients, dataset_path, gt_path, config,
                              shuffle=False, balanced=False, mu=mu, sigma=sigma)
    return train_loader, val_loader, test_loader


# ==========================================================
# Esecuzione di un fold
# ==========================================================
def run_fold(config, fold_idx, train_patients, val_patients, test_patients):
    print(f"\n===== FOLD {fold_idx+1} =====")
    train_loader, val_loader, test_loader = make_data_loaders_cv(config, train_patients, val_patients, test_patients)

    model = BIOTClassifier(
        n_channels=config["n_channels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    if config.get("pretrain_model_path", "") and config["finetune_mode"] in ["full_finetune", "frozen_encoder"]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_pretrained_encoder_into_biot(model, config["pretrain_model_path"], device)
        if config["finetune_mode"] == "frozen_encoder":
            for name, param in model.named_parameters():
                if name.startswith("biot."):
                    param.requires_grad = False
            print(" Encoder congelato.")

    lightning_model = LitModel_finetune(config, model)

    logger = TensorBoardLogger(save_dir="./", name="cv4fold_logs", version=f"fold{fold_idx+1}")
    ckpt = ModelCheckpoint(monitor="val_balanced_accuracy", mode="max", save_top_k=1, filename="best-model")
    early_stop = EarlyStopping(monitor="val_balanced_accuracy", mode="max", patience=config["early_stopping_patience"])

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[ckpt, early_stop],
        log_every_n_steps=50,
    )

    trainer.fit(lightning_model, train_loader, val_loader)
    val_metrics = trainer.validate(lightning_model, val_loader, ckpt_path="best")[0]
    test_metrics = trainer.test(lightning_model, test_loader, ckpt_path="best")[0]
    return val_metrics, test_metrics


# ==========================================================
# MAIN: 4-fold CV
# ==========================================================
if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    pl.seed_everything(42, workers=True)

    all_train = [f"chb{str(i).zfill(2)}" for i in range(1, 21)]
    test_patients = ["chb21", "chb22", "chb23"]
    folds = np.array_split(all_train, 4)

    all_val_metrics, all_test_metrics = [], []

    for fold_idx in range(4):
        val_patients = folds[fold_idx].tolist()
        train_patients = [p for p in all_train if p not in val_patients]
        val_m, test_m = run_fold(config, fold_idx, train_patients, val_patients, test_patients)
        all_val_metrics.append(val_m)
        all_test_metrics.append(test_m)

    # Calcolo media e std
    def mean_std(metrics_list, name):
        keys = metrics_list[0].keys()
        print(f"\n{name.upper()} - Mean ± Std:")
        summary = {}
        for k in keys:
            vals = [float(m[k]) for m in metrics_list if k in m]
            summary[k] = {"mean": np.mean(vals), "std": np.std(vals)}
            print(f"  {k:20s}: {summary[k]['mean']:.4f} ± {summary[k]['std']:.4f}")
        return summary

    val_summary = mean_std(all_val_metrics, "Validation (4-fold)")
    test_summary = mean_std(all_test_metrics, "Test (21–23)")

    import json
    with open("finetuning_cv4fold_summary.json", "w") as f:
        json.dump({"val": val_summary, "test": test_summary}, f, indent=4)

    print("\nRisultati salvati in finetuning_cv4fold_summary.json")