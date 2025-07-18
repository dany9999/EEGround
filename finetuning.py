# import os
# import pickle
# import torch
# from tqdm import tqdm
# import numpy as np
# import torch.nn as nn

# import pytorch_lightning as pl
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from model.SupervisedClassifier import BIOTClassifier
# from utils import CHBMITLoader, load_config
# from itertools import combinations

# from utils import BinaryBalancedAccuracy
# from torchmetrics.classification import (
#     BinaryAccuracy,
#     BinaryAveragePrecision,   # PR‑AUC
#     BinaryAUROC,               # ROC‑AUC
#     BinaryCohenKappa,
    
# )




# class LitModel_finetune(pl.LightningModule):
#     def __init__(self, config):
#         super().__init__()
        
#         self.config = config
#         self.save_hyperparameters()

#         self.model = BIOTClassifier()
#         self.criterion = nn.BCEWithLogitsLoss()

#         self.val_acc   = BinaryAccuracy()
#         self.val_prauc = BinaryAveragePrecision()
#         self.val_auroc = BinaryAUROC()
#         self.val_balacc = BinaryBalancedAccuracy()
#         self.val_CohenKappa = BinaryCohenKappa()

#         self.test_acc   = BinaryAccuracy()
#         self.test_prauc = BinaryAveragePrecision()
#         self.test_auroc = BinaryAUROC()
#         self.test_balacc = BinaryBalancedAccuracy()
#         self.test_CohenKappa = BinaryCohenKappa()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)
#         y = y.float().view(-1, 1)  # assicurati che y sia float e shape corretta
#         loss = self.criterion(logits, y)
#         self.log("train_loss", loss, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch

#         logits = self(x)


#         # Per la loss, y deve essere float
#         y_float = y.float().view(-1, 1)
#         loss = self.criterion(logits, y_float)
#         self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

#         # Probabilità per le metriche
#         probs = torch.sigmoid(logits)

#         # y deve essere int per torchmetrics
#         y_int = y.long().view(-1)

#         # Se probs è shape (batch_size, 1), rimuovi la seconda dimensione
#         probs = probs.view(-1)

#         self.val_acc.update(probs, y_int)
#         self.val_prauc.update(probs, y_int)
#         self.val_auroc.update(probs, y_int)
#         self.val_balacc.update(probs, y_int)
#         self.val_CohenKappa.update(probs, y_int)

#     def on_validation_epoch_end(self):
#         metrics = {
#             "val_acc": self.val_acc.compute(),
#             "val_pr_auc": self.val_prauc.compute(),
#             "val_auroc": self.val_auroc.compute(),
#             "val_bal_acc": self.val_balacc.compute(),
#             "val_cohen_kappa": self.val_CohenKappa.compute(),
#         }
#         self.log_dict(metrics, prog_bar=True, sync_dist=True)
#         self.val_acc.reset()
#         self.val_prauc.reset()
#         self.val_auroc.reset()
#         self.val_balacc.reset()
#         self.val_CohenKappa.reset()

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         logits = self(x)

#         # Per le metriche, y deve essere long (int64) 
#         y_int = y.long().view(-1)

#         # Probabilità
#         probs = torch.sigmoid(logits).view(-1)

#         self.test_acc.update(probs, y_int)
#         self.test_prauc.update(probs, y_int)
#         self.test_auroc.update(probs, y_int)
#         self.test_balacc.update(probs, y_int)
#         self.test_CohenKappa.update(probs, y_int)

#     def on_test_epoch_end(self):
#         metrics = {
#             "test_acc": self.test_acc.compute(),
#             "test_pr_auc": self.test_prauc.compute(),
#             "test_auroc": self.test_auroc.compute(),
#             "test_bal_acc": self.test_balacc.compute(),
#             "test_cohen_kappa": self.test_CohenKappa.compute(),
#         }
#         self.log_dict(metrics, prog_bar=True, sync_dist=True)
#         self.test_acc.reset()
#         self.test_prauc.reset()
#         self.test_auroc.reset()
#         self.test_balacc.reset()
#         self.test_CohenKappa.reset()

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(
#             self.model.parameters(), lr=self.config["lr"], weight_decay=float(self.config["weight_decay"])
#         )
#         return [optimizer]


# def leave_one_out_splits(patients, val_count=2):
#     splits = []
#     for i, test_patient in enumerate(patients):
#         remaining = [p for j, p in enumerate(patients) if j != i]
#         val_combinations = list(combinations(remaining, val_count))
#         for val_patients in val_combinations:
#             train_patients = [p for p in remaining if p not in val_patients]
#             splits.append({
#                 "train": train_patients,
#                 "val": list(val_patients),
#                 "test": [test_patient]
#             })
#             break  # solo la prima combinazione di validation
#     return splits


# def make_loader(patients_list, config):
#     segment_files = []
#     root = "CHB-MIT/clean_segments"
#     for patient in patients_list:
#         patient_path = os.path.join(root, patient)
#         if os.path.exists(patient_path):
#             files = [os.path.join(patient, f) for f in os.listdir(patient_path) if f.endswith(".pkl")]
#             segment_files.extend(files)
#     dataset = CHBMITLoader(root, segment_files, config["sampling_rate"])
#     return torch.utils.data.DataLoader(
#         dataset,
#         batch_size=config["batch_size"],
#         shuffle=True,
#         drop_last=True,
#         num_workers=config["num_workers"],
#         persistent_workers=True,
#     )


# def supervised(config, train_loader, val_loader, test_loader, iteration_idx):
#     lightning_model = LitModel_finetune(config)
#     os.makedirs("log-finetuning", exist_ok=True)

#     logger = TensorBoardLogger(save_dir="log-finetuning", name=f"run-{iteration_idx}")
#     early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    
    
#     checkpoint_callback = ModelCheckpoint(
#         dirpath=f"log-finetuning/run-{iteration_idx}/checkpoints",
#         filename="model-{epoch:02d}-{val_loss:.4f}",
#         monitor="val_loss",
#         mode="min",
#         save_last=True,
#         save_top_k=3,
#     )

#     trainer = pl.Trainer(
#         accelerator="cpu",
#         max_epochs=config["epochs"],
#         callbacks=[checkpoint_callback, early_stop_callback],
#         logger=logger,
#     )

#     dirpath_ckpt = f"log-finetuning/run-{iteration_idx}/checkpoints"
#     ckpt_path_last = os.path.join(dirpath_ckpt, "last.ckpt")
#     trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader , ckpt_path= ckpt_path_last)
    
    
#     result = trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")[0]
#     print(f"=== Split {iteration_idx} Result ===")
#     print(result)


# if __name__ == "__main__":
#     config = load_config("configs/finetuning.yml")
#     all_patients = sorted(os.listdir("CHB-MIT/clean_segments"))

#     splits = leave_one_out_splits(all_patients, val_count=2)

#     for idx, split in enumerate(splits):
#         print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")
#         train_loader = make_loader(split["train"], config)
#         val_loader = make_loader(split["val"], config)
#         test_loader = make_loader(split["test"], config)
#         supervised(config, train_loader, val_loader, test_loader, idx + 1)

import os
import pickle
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.SupervisedClassifier import BIOTClassifier
from utils import load_config, BinaryBalancedAccuracy
from CHBMITLoader import CHBMITLoader, make_loader
from itertools import combinations
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryCohenKappa,
)


class LitModelFinetune(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        self.model = BIOTClassifier()
        self.criterion = nn.BCEWithLogitsLoss()

        self.val_metrics = {
            "val_acc": BinaryAccuracy(),
            "val_pr_auc": BinaryAveragePrecision(),
            "val_auroc": BinaryAUROC(),
            "val_bal_acc": BinaryBalancedAccuracy(),
            "val_cohen_kappa": BinaryCohenKappa(),
        }

        self.test_metrics = {
            "test_acc": BinaryAccuracy(),
            "test_pr_auc": BinaryAveragePrecision(),
            "test_auroc": BinaryAUROC(),
            "test_bal_acc": BinaryBalancedAccuracy(),
            "test_cohen_kappa": BinaryCohenKappa(),
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = y.float().view(-1, 1)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y_float = y.float().view(-1, 1)
        loss = self.criterion(logits, y_float)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        probs = torch.sigmoid(logits).view(-1)
        y_int = y.long().view(-1)

        for metric in self.val_metrics.values():
            metric.update(probs, y_int)

    def on_validation_epoch_end(self):
        for name, metric in self.val_metrics.items():
            self.log(name, metric.compute(), prog_bar=True, sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits).view(-1)
        y_int = y.long().view(-1)

        for metric in self.test_metrics.values():
            metric.update(probs, y_int)

    def on_test_epoch_end(self):
        for name, metric in self.test_metrics.items():
            self.log(name, metric.compute(), prog_bar=True, sync_dist=True)
            metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], weight_decay=float(self.config["weight_decay"])
        )


def leave_one_out_splits(patients, val_count=2):
    splits = []
    for i, test_patient in enumerate(patients):
        remaining = [p for j, p in enumerate(patients) if j != i]
        val_combinations = list(combinations(remaining, val_count))
        val_patients = val_combinations[0]  # solo la prima combinazione
        train_patients = [p for p in remaining if p not in val_patients]
        splits.append({"train": train_patients, "val": list(val_patients), "test": [test_patient]})
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
    lightning_model = LitModelFinetune(config)

    logger = TensorBoardLogger(save_dir=config["log_dir"], name=f"run-{iteration_idx}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["save_dir"], f"run-{iteration_idx}"),
        filename="model-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        save_top_k=1,
        save_last=True,
        mode="min",
        every_n_epochs=config.get("save_every", 5),
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=config["early_stopping_patience"], mode="min")

    resume_ckpt = None
    if config.get("resume", False):
        resume_ckpt = os.path.join(config["save_dir"], f"run-{iteration_idx}", "last.ckpt")
        if not os.path.isfile(resume_ckpt):
            resume_ckpt = None

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,
        strategy="ddp_find_unused_parameters_false",
        max_epochs=config["epochs"],
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=logger,
        log_every_n_steps=10,
    )

    trainer.fit(lightning_model, train_loader, val_loader, ckpt_path=resume_ckpt)
    trainer.test(lightning_model, dataloaders=test_loader, ckpt_path="best")


if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    dataset_path = config["dataset_path"]
    #all_patients = sorted(os.listdir("CHB-MIT/clean_segments"))
    all_patients = sorted([p for p in os.listdir(dataset_path) if not p.startswith(".")])[:6]
    splits = leave_one_out_splits(all_patients, val_count=2)

    for idx, split in enumerate(splits):
        print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")
        train_loader = make_loader(split["train"], config)
        val_loader = make_loader(split["val"], config)
        test_loader = make_loader(split["test"], config)
        supervised(config, train_loader, val_loader, test_loader, idx + 1)