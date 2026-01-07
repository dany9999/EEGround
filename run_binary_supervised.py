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
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from model.SupervisedClassifier import BIOTClassifier
from utils import focal_loss, compute_global_stats, load_config

from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau, SequentialLR

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
# se CHBMITLoader è nella cartella padre
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from CHBMITLoader_4s import make_loader


def load_pretrained_encoder_into_biot(model, ckpt_path, device="cpu"):
    """
    Carica i pesi pretrainati (encoder_only_epoch_X.pt o model_epoch_X.pt) 
    nel modello BIOTClassifier, rinominando e filtrando automaticamente le chiavi.
    """


    print(f"\n Caricamento pesi pretrainati da: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Se è un checkpoint con struttura complessa (Lightning o torch.save(dict))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt

    # Normalizza chiavi (rimuove 'module.' se c'è)
    new_sd = {}
    for k, v in sd.items():
        k_clean = k[len("module."):] if k.startswith("module.") else k
        # Rimappa prefissi possibili
        if k_clean.startswith("encoder."):
            k_clean = re.sub(r"^encoder\.", "biot.", k_clean)
        elif not k_clean.startswith("biot."):
            k_clean = "biot." + k_clean
        new_sd[k_clean] = v

    model_dict = model.state_dict()
    compatible = {k: v for k, v in new_sd.items() if k in model_dict and v.shape == model_dict[k].shape}
    missing = set(model_dict.keys()) - set(compatible.keys())

    model_dict.update(compatible)
    model.load_state_dict(model_dict)
    print(f" Caricati {len(compatible)} layer nel backbone BIOT. "
          f"{len(missing)} layer mancanti (inizializzati da zero).")

    return model

def find_best_threshold(gt, probs, mode="f2"):
    """
    Trova la soglia ottimale in base a:
      - 'youden' → massimizza (TPR - FPR)
      - 'f1' → massimizza F1
      - 'f2' → massimizza F2 (più sensibile al recall)
    """
    gt = np.array(gt).astype(int)
    probs = np.array(probs).flatten()

    if np.sum(gt) == 0 or np.sum(gt) == len(gt):
        return 0.5, 0.0  # fallback

    if mode.lower() == "youden":
        fpr, tpr, ths = roc_curve(gt, probs)
        j_scores = tpr - fpr
        idx = np.argmax(j_scores)
        return float(ths[idx]), float(j_scores[idx])

    elif mode.lower() in ["f1", "f2"]:
        beta = 1.0 if mode.lower() == "f1" else 2.0
        prec, rec, ths = precision_recall_curve(gt, probs)
        f_scores = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-8)
        idx = np.nanargmax(f_scores)
        best_th = ths[idx] if idx < len(ths) else 0.5
        return float(best_th), float(f_scores[idx])

    else:
        return 0.5, 0.0

class LitModel_finetune(pl.LightningModule):
    def __init__(self, config, model):
        super().__init__()
        self.model = model
        self.threshold = config["threshold"]
        self.config = config
        self.alpha_focal = config["focal_alpha"]
        self.gamma_focal = config["focal_gamma"]
 
        self.criterion = nn.BCEWithLogitsLoss()

        # memorizza output per epoch
        self.val_results = {"preds": [], "targets": []}
        self.test_results = {"preds": [], "targets": []}



    def training_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        y = y.float().unsqueeze(1)
        logits = self.model(X)
        #loss = focal_loss(logits, y, alpha=self.alpha_focal, gamma=self.gamma_focal)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch["x"], batch["y"]
        y = y.float().unsqueeze(1) 
        with torch.no_grad():
            logits = self.model(X)
            #loss = focal_loss(logits, y, alpha=self.alpha_focal, gamma=self.gamma_focal)
            loss = self.criterion(logits, y)
            step_result = torch.sigmoid(logits).cpu().numpy()
            step_gt = y.cpu().numpy()
        self.val_results["preds"].append(step_result)
        self.val_results["targets"].append(step_gt)
        self.log("val_loss", loss)
        return loss


    def on_validation_epoch_end(self):
        result = np.concatenate(self.val_results["preds"])
        gt = np.concatenate(self.val_results["targets"])


        if sum(gt) * (len(gt) - sum(gt)) != 0:  # prevenzione AUROC error
            #self.threshold, score = find_best_threshold(gt, result, mode="f2")
            self.threshold =  np.sort(result)[-int(np.sum(gt))]
            #print(f"  [YOUDEN] Soglia ottimale trovata: {self.threshold:.4f} (score={score:.4f})")
            print(f"  Nuova soglia ottimale: {self.threshold}")

            results = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )

            # Calcolo sensitivity & specificity
            preds_bin = (result >= self.threshold).astype(int)
            
            
            cm = confusion_matrix(gt, preds_bin)
            tn, fp, fn, tp = cm.ravel()
            

            print("\nConfusion Matrix:")
            print(cm)
            print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)

            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
            }

        # Log metrics
        self.log("val_acc", results["accuracy"], sync_dist=True)
        self.log("val_bacc", results["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", results["pr_auc"], sync_dist=True)
        self.log("val_auroc", results["roc_auc"], sync_dist=True)
        self.log("val_sensitivity", results["sensitivity"], sync_dist=True)
        self.log("val_specificity", results["specificity"], sync_dist=True)

        print({
            k: float(v) for k, v in results.items()
        })

        # resetta buffer per epoch successivo
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
            results = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"],
                threshold=self.threshold,
            )

            # Calcolo sensitivity & specificity
            print("self.threshold:", self.threshold)
            preds_bin = (result >= self.threshold).astype(int)
            
            cm = confusion_matrix(gt, preds_bin)
            tn, fp, fn, tp = cm.ravel()
            
            
            print("\nConfusion Matrix:")
            print(cm)
            print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}\n")

            
            sensitivity = tp / (tp + fn + 1e-8)
            specificity = tn / (tn + fp + 1e-8)
    

            results["sensitivity"] = sensitivity
            results["specificity"] = specificity
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
            }

        # Log metrics
        self.log("test_acc", results["accuracy"], sync_dist=True)
        self.log("test_bacc", results["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", results["pr_auc"], sync_dist=True)
        self.log("test_auroc", results["roc_auc"], sync_dist=True)
        self.log("test_sensitivity", results["sensitivity"], sync_dist=True)
        self.log("test_specificity", results["specificity"], sync_dist=True)

        print({
            k: float(v) for k, v in results.items()
        })

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
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.config["lr"],
                weight_decay=float(self.config["weight_decay"]),
            )

        return optimizer
    







def predefined_split():
    
    # train_patients = [f"chb{str(i).zfill(2)}" for i in range(1, 20)]
    # val_patients   = [f"chb{str(i).zfill(2)}" for i in range(10, 20)]
    # test_patients  = [f"chb{str(i).zfill(2)}" for i in range(22, 24)]
    
    train_patients = ["chb01","chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", 
                      "chb10", "chb11", "chb12", "chb13", "chb14", "chb15","chb16", "chb17", "chb18", "chb19"]
     
    val_patients   = ["chb22", "chb23"]
    test_patients  = ["chb20", "chb21"]

    print("Train:", train_patients)
    print("Val:", val_patients)
    print("Test:", test_patients)

    return {"train": train_patients, "val": val_patients, "test": test_patients}

def prepare_CHB_MIT_dataloader(config):
    dataset_path = config["dataset_path_4s"]
    gt_path = "../../Datasets/chb_mit/GT"
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    split = predefined_split()

                  

    mu = np.load("global_mean.npy")
    sigma = np.load("global_std.npy")

    train_loader = make_loader(split["train"], dataset_path, gt_path, config,
                           shuffle=True, balanced=True, neg_to_pos_ratio=5, mu=mu, sigma=sigma)
    val_loader   = make_loader(split["val"], dataset_path, gt_path, config,
                           shuffle=False, mu=mu, sigma=sigma)
    test_loader  = make_loader(split["test"], dataset_path, gt_path, config,
                           shuffle=False, mu=mu, sigma=sigma)


    return train_loader, test_loader, val_loader




def supervised(config):
    train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(config)
    
    model = BIOTClassifier(
        n_channels=config["n_channels"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )

    #  Carica i pesi pretrained se specificato
    if config["finetune_mode"] == "full_finetune" or config["finetune_mode"] == "frozen_encoder":
        if config.get("pretrain_model_path", ""):
            ckpt_path = config["pretrain_model_path"]
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"\n Carico encoder pretrainato da {ckpt_path} su {device}")
            model = load_pretrained_encoder_into_biot(model, ckpt_path, device)

            
        # --- Freeza l'encoder (prima modifica singola) ---
        if config["finetune_mode"] == "frozen_encoder":
            for name, param in model.named_parameters():
                if name.startswith("biot."):
                    param.requires_grad = False
            print(" Encoder congelato: alleno solo la testa di classificazione")
    else:
        print(" Nessun modello pretrained specificato, pesi random.")


    

    lightning_model = LitModel_finetune(config, model)

    version = f"lr{config['lr']}-channels{config['n_channels']}-nfft{config['n_fft']}-hop{config['hop_length']}-frozen_encoder"
    #version = f"encLR{config['encoder_lr']:.1e}_headLR{config['head_lr']:.1e}-full_finetune"
    logger = TensorBoardLogger(save_dir="./", version=version, name=config["log_dir"])

    early_stop_callback = EarlyStopping(monitor="val_bacc", patience=config["early_stopping_patience"], verbose=False, mode="max")


    checkpoint_callback = ModelCheckpoint(
    monitor="val_bacc",
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
        log_every_n_steps=105, 
    )

  
   
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
# import pandas as pd
# import os

# def objective(trial):
#     # Carica config base
#     config = load_config("configs/finetuning.yml")

#     # Suggerisci iperparametri
    
#     config["encoder_lr"] = trial.suggest_loguniform("encoder_lr", 1e-7, 1e-5)
#     config["head_lr"] = trial.suggest_loguniform("head_lr", 1e-6, 1e-3)
    
    
#     config["epochs"] = 100

#     # Allena il modello e restituisci la metrica monitorata
#     results = supervised(config)
#     return results["val_bacc"]

# if __name__ == "__main__":
#     #  Usa storage persistente per poter riprendere dopo uno stop
#     storage_name = "sqlite:///optuna_finetuning_val_bacc.db"
#     study_name = "finetuning_tuning_val_bacc"

#     study = optuna.create_study(
#         study_name=study_name,
#         direction="maximize",
#         storage=storage_name,
#         load_if_exists=True,
#     )

#     #  Esegui l’ottimizzazione (puoi interrompere e riprendere)
#     study.optimize(objective, n_trials=15)

#     #  Stampa il risultato migliore
#     print("Best trial:")
#     trial = study.best_trial
#     print(f"  Value: {trial.value}")
#     for k, v in trial.params.items():
#         print(f"  {k}: {v}")

#     # Esporta risultati su CSV
#     df = study.trials_dataframe()

#     # Crea cartella risultati se non esiste
#     os.makedirs("results", exist_ok=True)
#     output_path = f"results/{study_name}_results.csv"
#     df.to_csv(output_path, index=False)
#     print(f"\n Risultati salvati in {output_path}")
