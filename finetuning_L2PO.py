import os
import argparse
import pickle
import sys
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import re
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pyhealth.metrics import binary_metrics_fn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from BIOT_vanilla.biot import BIOTClassifier
from utils import focal_loss, load_config

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CHBMITLoader_4s import make_loader


# =====================================================================
#                       LOAD PRETRAINED ENCODER
# =====================================================================
def load_pretrained_encoder_into_biot(model, ckpt_path, device="cpu"):
    print(f"\n Caricamento pesi pretrainati da: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    new_sd = {}
    for k, v in sd.items():
        k_clean = k[len("module."):] if k.startswith("module.") else k
        if k_clean.startswith("encoder."):
            k_clean = re.sub(r"^encoder\.", "biot.", k_clean)
        elif not k_clean.startswith("biot."):
            k_clean = "biot." + k_clean
        new_sd[k_clean] = v

    model_dict = model.state_dict()
    compatible = {k: v for k, v in new_sd.items()
                  if k in model_dict and v.shape == model_dict[k].shape}
    missing = set(model_dict.keys()) - set(compatible.keys())

    model_dict.update(compatible)
    model.load_state_dict(model_dict)

    print(f" Caricati {len(compatible)} layer nel backbone BIOT. "
          f"{len(missing)} layer mancanti (inizializzati da zero).")

    return model


# =====================================================================
#                      COMPUTE BEST THRESHOLD
# =====================================================================
def find_best_threshold(gt, probs, mode="f2"):
    gt = np.array(gt).astype(int)
    probs = np.array(probs).flatten()

    if np.sum(gt) == 0 or np.sum(gt) == len(gt):
        return 0.5, 0.0

    if mode.lower() in ["f1", "f2"]:
        beta = 1.0 if mode.lower() == "f1" else 2.0
        prec, rec, ths = precision_recall_curve(gt, probs)
        f_scores = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-8)
        idx = np.nanargmax(f_scores)
        best_th = ths[idx] if idx < len(ths) else 0.5
        return float(best_th), float(f_scores[idx])

    fpr, tpr, ths = roc_curve(gt, probs)
    j_scores = tpr - fpr
    idx = np.argmax(j_scores)
    return float(ths[idx]), float(j_scores[idx])


# =====================================================================
#                  LIGHTNING FINE-TUNING MODULE
# =====================================================================
class LitModel_finetune(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.threshold = config["threshold"]
        self.config = config

        self.criterion = nn.BCEWithLogitsLoss()
        self.val_results = {"preds": [], "targets": []}
        self.test_results = {"preds": [], "targets": []}

    def training_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        y = y.float().unsqueeze(1)
        logits = self.model(X)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        y = y.float().unsqueeze(1)

        with torch.no_grad():
            logits = self.model(X)
            loss = self.criterion(logits, y)
            preds = torch.sigmoid(logits).cpu().numpy()

        self.val_results["preds"].append(preds)
        self.val_results["targets"].append(y.cpu().numpy())
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.val_results["preds"])
        gt = np.concatenate(self.val_results["targets"])

        if np.sum(gt) not in [0, len(gt)]:
            self.threshold, _ = find_best_threshold(gt, preds, mode="f2")
            results = binary_metrics_fn(gt, preds,
                                        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                                        threshold=self.threshold)

            preds_bin = (preds >= self.threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(gt, preds_bin).ravel()

            results["sensitivity"] = tp / (tp + fn + 1e-8)
            results["specificity"] = tn / (tn + fp + 1e-8)
        else:
            results = {m: 0.0 for m in ["accuracy","balanced_accuracy","pr_auc","roc_auc","sensitivity","specificity"]}

        for k, v in results.items():
            self.log(f"val_{k}", v, sync_dist=True)

        self.val_results = {"preds": [], "targets": []}

    def test_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        with torch.no_grad():
            logits = self.model(X)
            preds = torch.sigmoid(logits).cpu().numpy()
            targets = y.cpu().numpy()
        self.test_results["preds"].append(preds)
        self.test_results["targets"].append(targets)

    def on_test_epoch_end(self):
        preds = np.concatenate(self.test_results["preds"])
        gt = np.concatenate(self.test_results["targets"])

        if np.sum(gt) not in [0, len(gt)]:
            results = binary_metrics_fn(gt, preds,
                                        metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                                        threshold=self.threshold)

            preds_bin = (preds >= self.threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(gt, preds_bin).ravel()

            results["sensitivity"] = tp / (tp + fn + 1e-8)
            results["specificity"] = tn / (tn + fp + 1e-8)
        else:
            results = {m: 0.0 for m in ["accuracy","balanced_accuracy","pr_auc","roc_auc","sensitivity","specificity"]}

        for k, v in results.items():
            self.log(f"test_{k}", v, sync_dist=True)

        self.test_results = {"preds": [], "targets": []}


    def configure_optimizers(self):
        if self.config["finetune_mode"] == "full_finetune":
            enc_params = [p for n, p in self.named_parameters() if "biot" in n]
            head_params = [p for n, p in self.named_parameters() if "biot" not in n]
            return torch.optim.Adam(
                [{"params": enc_params, "lr": self.config["encoder_lr"]},
                 {"params": head_params, "lr": self.config["head_lr"]}],
                weight_decay=self.config["weight_decay"]
            )

        return torch.optim.Adam(
            self.parameters(),
            lr=self.config["lr"],
            weight_decay=self.config["weight_decay"]
        )


# =====================================================================
#                  SPLIT: VAL FISSO + 10 RUN TEST
# =====================================================================
def predefined_split(run_id=1):

    all_subjects = [f"chb{str(i).zfill(2)}" for i in range(1, 24)]
    val = ["chb20", "chb21", "chb22"]

    test_pairs = [
        ["chb01", "chb02"], ["chb03", "chb04"], ["chb05", "chb06"],
        ["chb07", "chb08"], ["chb09", "chb10"], ["chb11", "chb12"],
        ["chb13", "chb14"], ["chb15", "chb16"], ["chb17", "chb18"],
        ["chb19", "chb23"],
    ]

    if not (1 <= run_id <= 10):
        raise ValueError("run_id deve essere tra 1 e 10")

    test = test_pairs[run_id - 1]
    train = [s for s in all_subjects if s not in val + test]

    print("\n================ RUN", run_id, "================")
    print("Validation:", val)
    print("Test:", test)
    print("Train:", train)
    print("==============================================\n")

    return {"train": train, "val": val, "test": test}


# =====================================================================
#                      DATA LOADERS
# =====================================================================
def prepare_CHB_MIT_dataloader(config, run_id=1):
    dataset_path = config["dataset_path_4s"]
    gt_path = "../../Datasets/chb_mit/GT"

    split = predefined_split(run_id)
    mu = np.load("global_mean.npy")
    sigma = np.load("global_std.npy")

    train_loader = make_loader(split["train"], dataset_path, gt_path, config,
                               shuffle=True, balanced=True, neg_to_pos_ratio=5,
                               mu=mu, sigma=sigma)

    val_loader = make_loader(split["val"], dataset_path, gt_path, config,
                             shuffle=False, mu=mu, sigma=sigma)

    test_loader = make_loader(split["test"], dataset_path, gt_path, config,
                              shuffle=False, mu=mu, sigma=sigma)

    return train_loader, test_loader, val_loader


# =====================================================================
#                       SUPERVISED FINETUNING
# =====================================================================
def supervised(config, run_id=1):
    train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(config, run_id)

    model = BIOTClassifier(
        n_channels=config["n_channels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    if config["finetune_mode"] in ["full_finetune", "frozen_encoder"]:
        if config.get("pretrain_model_path", ""):
            model = load_pretrained_encoder_into_biot(
                model, config["pretrain_model_path"],
                device=torch.device("cuda:0")
            )

        if config["finetune_mode"] == "frozen_encoder":
            for name, param in model.named_parameters():
                if name.startswith("biot."):
                    param.requires_grad = False

    lightning_model = LitModel_finetune(config, model)

    logger = TensorBoardLogger(
        save_dir="./",
        version=f"run{run_id}",
        name=config["log_dir"]
    )

    early_stop_callback = EarlyStopping(
        monitor="val_bacc",
        patience=config["early_stopping_patience"],
        mode="max"
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_bacc",
        save_top_k=1,
        mode="max",
        filename=f"best-model-run{run_id}"
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        logger=logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=config["epochs"],
        benchmark=True,
    )

    trainer.fit(lightning_model, train_loader, val_loader)

    val_metrics = trainer.validate(lightning_model, val_loader, ckpt_path="best")[0]
    test_metrics = trainer.test(lightning_model, test_loader, ckpt_path="best")[0]

    return val_metrics, test_metrics


# =====================================================================
#                              MAIN
# =====================================================================
if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")

    all_val = []
    all_test = []

    for run_id in range(1, 11):
        print(f"\n\n================= RUN {run_id} =================\n")
        val_m, test_m = supervised(config, run_id=run_id)
        all_val.append(val_m)
        all_test.append(test_m)

    print("\n\n=========== MEDIA E STD SU 10 RUN ===========")

    def mean_std_dict(list_of_dicts):
        keys = list_of_dicts[0].keys()
        out = {}
        for k in keys:
            vals = np.array([float(d[k]) for d in list_of_dicts])
            out[k] = {"mean": float(vals.mean()), "std": float(vals.std())}
        return out

    val_stats = mean_std_dict(all_val)
    test_stats = mean_std_dict(all_test)

    print("\nValidation Mean ± Std:\n", val_stats)
    print("\nTest Mean ± Std:\n", test_stats)

    import json
    summary = {"val_runs": all_val, "test_runs": all_test,
               "val_mean_std": val_stats, "test_mean_std": test_stats}

    with open("finetuning_results_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\nRisultati salvati in finetuning_results_summary.json")