import os
import argparse

import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model.SelfSupervisedPretrain import UnsupervisedPretrain
from preprocessing import percentile_95_normalize

from utils import load_config, EEGDataset
from torch.utils.data import DataLoader, random_split



class LitModel_self_supervised_pretrain(pl.LightningModule):
    def __init__(self, config, save_path):
        super().__init__()
        self.config = config
        self.save_path = save_path
        self.model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=23) 
        
    def training_step(self, batch, batch_idx):
        # Salvataggio del checkpoint ogni N passi
      

        samples = batch  # [B, C, T]
        # Normalizza i campioni
        samples = percentile_95_normalize(samples)  # Normalizzazione al 95° percentile 
        original , mask, reconstruction = self.model(samples) 

        # Calcola la MSE solo sulle posizioni mascherate
        loss = F.mse_loss(reconstruction[mask], original[mask])

        self.log("train_loss", loss)
        return loss
    

    def validation_step(self, batch, batch_idx):
        
        samples = batch  # [B, C, T]
        # Normalizza i campioni
        samples = percentile_95_normalize(samples)  # Normalizzazione al 95° percentile 
        original , mask, reconstruction = self.model(samples) 

        # Calcola la MSE solo sulle posizioni mascherate
        loss = F.mse_loss(reconstruction[mask], original[mask])

        self.log("val_loss", loss)
        return loss




    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config["lr"], weight_decay=float(self.config["weight_decay"])
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )

        return [optimizer], [scheduler]
    




def prepare_dataloader_TUAB(config):
    # Percorsi a entrambe le cartelle

    abnormal_dir = os.path.abspath(os.path.join("..", "..", "Datasets/TUH/TUAB/Abnormal/REF"))
    normal_dir = os.path.abspath(os.path.join("..", "..", "Datasets/TUH/TUAB/Normal/REF"))

    dataset = EEGDataset([abnormal_dir, normal_dir])

    # Split 70/30
    total_len = len(dataset)
    val_len = int(0.3 * total_len)
    train_len = total_len - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        persistent_workers=True,
        drop_last=False,
    )

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    return train_loader, val_loader
   


def pretrain(config):
    


    # get data loaders
    train_loader, valid_loader = prepare_dataloader_TUAB(config)
     
    
   
    os.makedirs("log-pretrain", exist_ok=True)
   
    
    # Definizione del path per il salvataggio
    output_dir = "log-pretrain"
    save_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(save_path, exist_ok=True)
    
    # define the model
    model = LitModel_self_supervised_pretrain(config, save_path)
    
    
    # Checkpoint dei 3 migliori modelli + l'ultimo (basato su val_loss)
    best_ckpt = ModelCheckpoint(
        dirpath=os.path.join(save_path, "best"),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )

    # Checkpoint frequenti: salva ogni 200 step, per recuperare se crasha
    step_ckpt = ModelCheckpoint(
        dirpath=os.path.join(save_path, "step"),
        filename="step-{step}",
        every_n_train_steps=2000,
        save_top_k=-1  # salva tutti i checkpoint di step
    )

  
    # define the logger
    logger = TensorBoardLogger(save_dir=output_dir, name="logs")



    #trainer in distributed mode
    # trainer = pl.Trainer(
    #     devices=[1],
    #     accelerator="gpu",
    #     benchmark=True,
    #     #strategy=DDPStrategy(find_unused_parameters=False),
    #     #auto_select_gpus=True,
    #     enable_checkpointing=True,
    #     logger=logger,
    #     callbacks=[best_ckpt, step_ckpt],
    #     max_epochs=config["epochs"]
    # )


    #trainer cpu
    trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=config["epochs"],
    enable_checkpointing=True,
    callbacks=[best_ckpt, step_ckpt],
    logger=logger,
    )


    # train the model
    trainer.fit(model, train_loader, valid_loader, ckpt_path="last")




if __name__ == "__main__":

    config = load_config("configs/pretraining.yml")
    
   
    print (config)

    pretrain(config)    