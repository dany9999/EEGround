import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pyhealth.metrics import binary_metrics_fn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, average_precision_score
import random
from biot import BIOTClassifier
from utils import focal_loss, compute_global_stats, load_config

from ..CHBMITLoader import make_loader
class LitModel_finetune(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.config = config

    def compute_binary_metrics(y_true, y_pred, threshold=0.5):
        y_pred_bin = (y_pred >= threshold).astype(int)
        
        metrics = {}
        metrics["accuracy"] = accuracy_score(y_true, y_pred_bin)
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred_bin)
        
        # roc_auc_score e average_precision_score richiedono almeno una classe positiva e una negativa
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred)
            metrics["pr_auc"] = average_precision_score(y_true, y_pred)
        else:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
        
        return metrics

    def training_step(self, batch, batch_idx):
        X, y = batch
        prob = self.model(X)
        loss = focal_loss(prob, y)  # focal_loss(prob, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def validation_epoch_end(self, val_step_outputs):
        result = np.array([])
        gt = np.array([])
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            # result = binary_metrics_fn(
            #     gt,
            #     result,
            #     metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
            #     threshold=self.threshold,
            # )
            result = self.compute_binary_metrics(gt, result, threshold=self.threshold)
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        print(result)

    def test_step(self, batch, batch_idx):
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        return step_result, step_gt

    def test_epoch_end(self, test_step_outputs):
        result = np.array([])
        gt = np.array([])
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            # result = binary_metrics_fn(
            #     gt,
            #     result,
            #     metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
            #     threshold=self.threshold,
            # )
            result = self.compute_binary_metrics(gt, result, threshold=self.threshold)
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)

        return result

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"],
        )

        return [optimizer]  # , [scheduler]


def predefined_split():
    train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    val_patients = [f"chb{str(i).zfill(2)}" for i in range(20, 22)]
    test_patients = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    return {"train": train_patients, "val": val_patients, "test": test_patients}



def prepare_CHB_MIT_dataloader(config):
    dataset_path = config["dataset_path"]
    gt_path = "../../Datasets/chb_mit/GT"
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    split = predefined_split()

    train_mean, train_std = compute_global_stats(split["train"], dataset_path)
    mean_t = torch.tensor(train_mean, dtype=torch.float32).view(18, 1)
    std_t = torch.tensor(train_std, dtype=torch.float32).view(18, 1)

    train_loader = make_loader(split["train"], dataset_path, gt_path, config, mean_t, std_t, balanced=True, shuffle=True)
    val_loader   = make_loader(split["val"], dataset_path, gt_path, config, mean_t, std_t, shuffle=False)
    test_loader  = make_loader(split["test"], dataset_path, gt_path, config, mean_t, std_t, shuffle=False)
    return train_loader, test_loader, val_loader





def supervised(config):



    train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(config)
    model = BIOTClassifier(
        
        # set the n_channels according to the pretrained model if necessary
        n_channels= config["n_channels"],
        n_fft= 200,
        hop_length= 100,
    )

    lightning_model = LitModel_finetune(config, model)

    # logger and callbacks
    version = f"CHB-MIT-{config["lr"]}-{config["batch_size"]}"
    logger = TensorBoardLogger(
        save_dir="./",
        version=version,
        name="log",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_auroc", patience=5, verbose=False, mode="max"
    )

    trainer = pl.Trainer(
        devices=[0],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=config["epochs"],
        callbacks=[early_stop_callback],
    )

    # train the model
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )

    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path="best", dataloaders=test_loader
    )[0]
    print(pretrain_result)


if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")

    supervised(config)
