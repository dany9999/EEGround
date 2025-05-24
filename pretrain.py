import os
import argparse
import pickle

import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import UnsupervisedPretrain





class LitModel_self_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
        self.T = 0.2
        self.model = UnsupervisedPretrain(emb_size=256, heads=8, depth=4, n_channels=18) 
        
    def training_step(self, batch, batch_idx):
        # Salvataggio del checkpoint ogni N passi
        if self.global_step % 2000 == 0:
            self.trainer.save_checkpoint(
                filepath=f"{self.save_path}/epoch={self.current_epoch}_step={self.global_step}.ckpt"
            )

        samples = batch  # [B, C, T] 
        original , mask, reconstruction = self.model(samples) 

        # Calcola la MSE solo sulle posizioni mascherate
        loss = F.mse_loss(reconstruction[mask], original[mask])

        self.log("train_loss", loss)
        return loss
    



    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay
        )

        # set learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.3
        )

        return [optimizer], [scheduler]
    


def prepare_dataloader(args):
   pass
   
   

def pretrain(args):
    
    # get data loaders
    train_loader = prepare_dataloader(args)
    
    # define the trainer
    N_version = (
        len(os.listdir(os.path.join("log-pretrain"))) + 1
    )
    # define the model
    save_path = f"log-pretrain/{N_version}-unsupervised/checkpoints"
    
    model = LitModel_supervised_pretrain(args, save_path)
    
    logger = TensorBoardLogger(
        save_dir="/home/chaoqiy2/github/LEM",
        version=f"{N_version}/checkpoints",
        name="log-pretrain",
    )
    trainer = pl.Trainer(
        devices=[2],
        accelerator="gpu",
        strategy=DDPStrategy(find_unused_parameters=False),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
        logger=logger,
        max_epochs=args.epochs,
    )

    # train the model
    trainer.fit(model, train_loader)

    