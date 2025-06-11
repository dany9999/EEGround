import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,   # PR‑AUC
    BinaryAUROC               # ROC‑AUC
)

from utils import load_config
from model.SupervisedClassifier import BIOTClassifier
from utils import CHBMITLoader
from itertools import combinations


class LitModel_finetune(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.model = BIOTClassifier()
        self.config = config

        self.criterion = nn.BCEWithLogitsLoss()

        self.val_acc   = BinaryAccuracy(threshold=0.5)
        self.val_prauc = BinaryAveragePrecision()
        self.val_auroc = BinaryAUROC()

        self.test_acc   = BinaryAccuracy(threshold=0.5)
        self.test_prauc = BinaryAveragePrecision()
        self.test_auroc = BinaryAUROC()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float().view(-1, 1)  # assicurati che y sia float e shape corretta
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        print("logit shape:", logits.shape)
        print("y shape:", y.shape)
        y = y.float().view(-1, 1)
        loss = self.criterion(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        probs = torch.sigmoid(logits)
        self.val_acc.update(probs, y)
        self.val_prauc.update(probs, y)
        self.val_auroc.update(probs, y)

    def on_validation_epoch_end(self):
        metrics = {
            "val_acc": self.val_acc.compute(),
            "val_pr_auc": self.val_prauc.compute(),
            "val_auroc": self.val_auroc.compute(),
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.val_acc.reset()
        self.val_prauc.reset()
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float().view(-1, 1)
        probs = torch.sigmoid(logits)
        self.test_acc.update(probs, y)
        self.test_prauc.update(probs, y)
        self.test_auroc.update(probs, y)

    def on_test_epoch_end(self):
        metrics = {
            "test_acc": self.test_acc.compute(),
            "test_pr_auc": self.test_prauc.compute(),
            "test_auroc": self.test_auroc.compute(),
        }
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        self.test_acc.reset()
        self.test_prauc.reset()
        self.test_auroc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], weight_decay=float(self.config["weight_decay"])
        )
        return [optimizer]


def leave_one_out_splits(patients, val_count=2):
    splits = []
    for i, test_patient in enumerate(patients):
        remaining = [p for j, p in enumerate(patients) if j != i]
        val_combinations = list(combinations(remaining, val_count))
        for val_patients in val_combinations:
            train_patients = [p for p in remaining if p not in val_patients]
            splits.append({
                "train": train_patients,
                "val": list(val_patients),
                "test": [test_patient]
            })
            break  # solo la prima combinazione di validation
    return splits


def make_loader(patients_list, config):
    segment_files = []
    root = "CHB-MIT/clean_segments"
    for patient in patients_list:
        patient_path = os.path.join(root, patient)
        if os.path.exists(patient_path):
            files = [os.path.join(patient, f) for f in os.listdir(patient_path) if f.endswith(".pkl")]
            segment_files.extend(files)
    dataset = CHBMITLoader(root, segment_files, config["sampling_rate"])
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
    )


def supervised(config, train_loader, val_loader, test_loader, iteration_idx):
    lightning_model = LitModel_finetune(config)
    os.makedirs("log-finetuning", exist_ok=True)

    logger = TensorBoardLogger(save_dir="log-finetuning", name=f"run-{iteration_idx}")
    early_stop_callback = EarlyStopping(monitor="val_auroc", patience=5, mode="max")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"log-finetuning/run-{iteration_idx}/checkpoints",
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=3,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
    )

    trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    result = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")[0]
    print(f"=== Split {iteration_idx} Result ===")
    print(result)


if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    all_patients = sorted(os.listdir("CHB-MIT/clean_segments"))

    splits = leave_one_out_splits(all_patients, val_count=2)

    for idx, split in enumerate(splits):
        print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")
        train_loader = make_loader(split["train"], config)
        val_loader = make_loader(split["val"], config)
        test_loader = make_loader(split["test"], config)
        supervised(config, train_loader, val_loader, test_loader, idx + 1)