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

from model.BIOT import UnsupervisedPretrain
from utils import UnsupervisedPretrainLoader





class LitModel_self_supervised_pretrain(pl.LightningModule):
    def __init__(self, args, save_path):
        super().__init__()
        self.args = args
        self.save_path = save_path
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
    

    def validation_step(self, batch, batch_idx):
        
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
    


def prepare_dataloader_train(args):
    # Imposta il seed per la riproducibilità
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Percorso al file numpy
    npy_path = "./CHB-MIT/train_numpy/all_segments.npy"
    data = np.load(npy_path)  # shape: (N, C, T)
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Crea un dataset PyTorch
    dataset = torch.utils.data.TensorDataset(tensor_data)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=True,
    )

    return train_loader


def prepare_dataloader_validation(args):
    # Imposta il seed per la riproducibilità
    seed = 12345
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Percorso al file numpy di validazione
    npy_path = "./CHB-MIT/validation_numpy/all_segments.npy"
    data = np.load(npy_path)  # shape: (N, C, T)
    tensor_data = torch.tensor(data, dtype=torch.float32)

    # Crea un dataset PyTorch
    dataset = torch.utils.data.TensorDataset(tensor_data)

    # DataLoader
    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        drop_last=False,
    )

    return val_loader
   
   

def pretrain(args):
    


    # get data loaders
    train_loader = prepare_dataloader_train(args)
    valid_loader = prepare_dataloader_validation(args)

    
    # define the trainer
    os.makedirs("log-pretrain", exist_ok=True)
    N_version = (
        len(os.listdir(os.path.join("log-pretrain"))) + 1
    )
    # define the model
    save_path = f"log-pretrain/{N_version}-unsupervised/checkpoints"
    os.makedirs(save_path, exist_ok=True)
    
    model = LitModel_self_supervised_pretrain(args, save_path)
    
    logger = TensorBoardLogger(
        save_dir="log-pretrain",
        version=f"{N_version}/checkpoints",
        name="unsupervised",
    )

    # trainer in distributed mode
    # trainer = pl.Trainer(
    #     devices=[1],
    #     accelerator="cpu",
    #     strategy=DDPStrategy(find_unused_parameters=False),
    #     #auto_select_gpus=True,
    #     benchmark=True,
    #     enable_checkpointing=True,
    #     logger=logger,
    #     max_epochs=args.epochs,
    # )

    # trainer cpu
    trainer = pl.Trainer(
    accelerator="cpu",
    max_epochs=args.epochs,
    enable_checkpointing=True,
    logger=logger,
    )



    # train the model
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    args = parser.parse_args()
    print (args)

    pretrain(args)    