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

# se CHBMITLoader è nella cartella padre
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CHBMITLoader import make_loader, EEGAugment


class LitModel_finetune(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.threshold = config["threshold"]
        self.config = config
        self.alpha_focal = config["focal_alpha"]
        self.gamma_focal = config["focal_gamma"]


        # memorizza output per epoch
        self.val_results = {"preds": [], "targets": []}
        self.test_results = {"preds": [], "targets": []}



    def training_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        prob = self.model(X)
        loss = focal_loss(prob, y, alpha=self.alpha_focal, gamma=self.gamma_focal)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.val_results["preds"].append(step_result)
        self.val_results["targets"].append(step_gt)


    def on_validation_epoch_end(self):
        result = np.concatenate(self.val_results["preds"])
        gt = np.concatenate(self.val_results["targets"])

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            results = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("val_acc", results["accuracy"], sync_dist=True)
        self.log("val_bacc", results["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", results["pr_auc"], sync_dist=True)
        self.log("val_auroc", results["roc_auc"], sync_dist=True)
        print(results)
        # resetta per il prossimo epoch
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

        if (
            sum(gt) * (len(gt) - sum(gt)) != 0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
        self.log("test_acc", results["accuracy"], sync_dist=True)
        self.log("test_bacc", results["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", results["pr_auc"], sync_dist=True)
        self.log("test_auroc", results["roc_auc"], sync_dist=True)

        self.test_results = {"preds": [], "targets": []}
        return results

    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config["lr"],
            weight_decay=float(self.config["weight_decay"]),
        )
        return [optimizer]


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

    #train_mean, train_std = compute_global_stats(split["train"], dataset_path)
    #mean_t = torch.tensor(train_mean, dtype=torch.float32).view(18, 1)
    #std_t = torch.tensor(train_std, dtype=torch.float32).view(18, 1)

    
    augment_pos = EEGAugment(p_jitter=0.7, p_scale=0.5, p_mask=0.3, jitter_std=0.02)
    
    train_loader = make_loader(split["train"], dataset_path, gt_path, config,
                               shuffle=True, balanced=False,
                               pos_oversample_k=4, transform=augment_pos,
                               neg_undersample_ratio=0.3)  # <-- tieni solo 30% dei negativi

    val_loader   = make_loader(split["val"], dataset_path, gt_path, config,
                               shuffle=False, balanced=False,
                               pos_oversample_k=0, transform=None)

    test_loader  = make_loader(split["test"], dataset_path, gt_path, config,
                               shuffle=False, balanced=False,
                               pos_oversample_k=0, transform=None)
    


    return train_loader, test_loader, val_loader




def supervised(config):
    train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(config)
    
    model = BIOTClassifier(
        n_channels=config["n_channels"],
        n_fft=200,
        hop_length=100,
    )

    # # 🔹 Caricamento pesi pretrained se specificato
    # if config.get("pretrain_model_path", ""):
    #     state = torch.load(config["pretrain_model_path"], map_location="cpu")
    #     model_dict = model.biot.state_dict()

    #     # allinea i layer con la stessa shape
    #     compatible_state = {k: v for k, v in state.items()
    #                         if k in model_dict and v.shape == model_dict[k].shape}
    #     missing = set(model_dict.keys()) - set(compatible_state.keys())
    #     print(f"Carico {len(compatible_state)} layer dai pretrained, "
    #           f"{len(missing)} inizializzati random.")

    #     # aggiorna i pesi
    #     model_dict.update(compatible_state)
    #     model.biot.load_state_dict(model_dict)


    lightning_model = LitModel_finetune(config, model)

    version = f"CHB-MIT-{config["finetune_mode"]}"
    logger = TensorBoardLogger(save_dir="./", version=version, name="log")

    early_stop_callback = EarlyStopping(monitor="val_pr_auc", patience=15, verbose=False, mode="max")


    checkpoint_callback = ModelCheckpoint(
    monitor="val_pr_auc",
    mode="max",
    save_top_k=1,
    filename="best-model"
    )

    trainer = pl.Trainer(
        devices= 1,
        accelerator="gpu",
        strategy= "auto", #DDPStrategy(find_unused_parameters=False),
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=config["epochs"],
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # trainer.fit(lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # pretrain_result = trainer.test(model=lightning_model, dataloaders=test_loader, ckpt_path="best")[0]
    # print(pretrain_result)
   
    trainer.fit(lightning_model, train_loader, val_loader)
    val_metrics = trainer.validate(model=lightning_model, dataloaders=val_loader, ckpt_path="best")[0]




    # test (solo per log)
    test_results = trainer.test(model=lightning_model, dataloaders=test_loader, ckpt_path="best")[0]
    print("Test results:", test_results)

    return val_metrics

 


if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    supervised(config)

# import optuna

# def objective(trial):
#     # Carica config base
#     config = load_config("configs/finetuning.yml")

#     # Suggerisci iperparametri
#     config["lr"] = trial.suggest_loguniform("lr", 1e-6, 1e-3)
#     config["focal_alpha"] = trial.suggest_uniform("focal_alpha", 0.1, 0.9)
#     config["focal_gamma"] = trial.suggest_uniform("focal_gamma", 0.5, 3.0)
#     #config["threshold"] = trial.suggest_uniform("threshold", 0.1, 0.9)

#     # Limita epoche per tuning veloce
#     config["epochs"] = 60

#     # Allena e ottieni risultati
#     results = supervised(config)  # deve ritornare i risultati
#     return results["val_pr_auc"]  # metriche monitorate

# if __name__ == "__main__":
#     study = optuna.create_study(direction="maximize")
#     study.optimize(objective, n_trials=10)

#     print("Best trial:")
#     trial = study.best_trial
#     print(f"Value: {trial.value}")
#     print("Params:")
#     for k, v in trial.params.items():
#         print(f"  {k}: {v}")