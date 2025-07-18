
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import combinations
from torch.utils.tensorboard import SummaryWriter
from CHBMITLoader import make_loader
from model.SupervisedClassifier import BIOTClassifier
from utils import load_config, BinaryBalancedAccuracy
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryCohenKappa
)
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def leave_one_out_splits(patients, val_count=2):
    splits = []
    for i, test_patient in enumerate(patients):
        remaining = [p for j, p in enumerate(patients) if j != i]
        val_combinations = list(combinations(remaining, val_count))
        for val_patients in val_combinations:
            train_patients = [p for p in remaining if p not in val_patients]
            splits.append({"train": train_patients, "val": list(val_patients), "test": [test_patient]})
            break
    return splits


def run_epoch(model, dataloader, criterion, optimizer, device, mode, metrics, writer=None, global_step=0):
    model.train() if mode == "train" else model.eval()
    running_loss = 0.0

    for metric in metrics.values():
        metric.reset()

    per_file_preds = {}

    for batch in tqdm(dataloader, desc=mode.capitalize()):
        if isinstance(batch, dict):
            x = batch["x"].to(device)
            y = batch["y"].to(device).float().view(-1, 1)
            file_ids = batch["file"]

        with torch.set_grad_enabled(mode == "train"):
            logits = model(x)
            loss = criterion(logits, y)
            if mode == "train":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item()
        probs = torch.sigmoid(logits).view(-1)
        y_int = y.long().view(-1)

        for m in metrics.values():
            m.update(probs, y_int)

        if mode == "test":
            for f_id, p, t in zip(file_ids, probs.detach().cpu(), y_int.cpu()):
                if f_id not in per_file_preds:
                    per_file_preds[f_id] = {"y_true": [], "y_pred": []}
                per_file_preds[f_id]["y_true"].append(t.item())
                per_file_preds[f_id]["y_pred"].append(int(p >= 0.5))

    avg_loss = running_loss / len(dataloader)
    return avg_loss, global_step, per_file_preds if mode == "test" else None


def compute_metrics(metrics):
    results = {}
    for name, metric in metrics.items():
        val = metric.compute()
        if hasattr(val, 'item'):
            val = val.item()
        results[name] = val
    for metric in metrics.values():
        metric.reset()
    return results


def supervised(config, train_loader, val_loader, test_loader, iteration_idx):
    device, local_rank = setup_ddp()

    model = BIOTClassifier(
        emb_size=config['emb_size'],
        heads=config['heads'],
        depth=config['depth'],
        n_classes=config['n_classes']
    ).to(device)

    finetune_mode = config["finetune_mode"]
    optimizer = None

    if finetune_mode in ["frozen_encoder", "full_finetune"]:
        checkpoint = torch.load(config["pretrained_ckpt"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if finetune_mode == "frozen_encoder":
        for param in model.biot.parameters():
            param.requires_grad = False
        optimizer = optim.Adam(model.classifier.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    elif finetune_mode == "full_finetune":
        for param in model.biot.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    elif finetune_mode == "from_scratch":
        model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
        optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}")

    model = DDP(model, device_ids=[local_rank])
    criterion = nn.BCEWithLogitsLoss()

    test_metrics = {
        "acc": BinaryAccuracy().to(device),
        "prauc": BinaryAveragePrecision().to(device),
        "auroc": BinaryAUROC().to(device),
        "balacc": BinaryBalancedAccuracy().to(device),
        "kappa": BinaryCohenKappa().to(device)
    }

    log_dir = f"{config.get('log_dir', 'log-finetuning')}/run-{iteration_idx}-{finetune_mode}"
    writer = SummaryWriter(log_dir=log_dir) if local_rank == 0 else None

    best_model_path = config['save_dir'].format(iteration_idx=iteration_idx, mode=finetune_mode)
    ckpt_path = best_model_path + ".ckpt"

    start_epoch = 0
    best_val_loss = float("inf")
    counter = 0
    global_step = 0

    resume = config["resume"]
    if resume and os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        counter = checkpoint["counter"]
        global_step = checkpoint.get("global_step", 0)

    patience = config["early_stopping_patience"]
    save_every = config["save_every"]

    for epoch in range(start_epoch, config["epochs"]):
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, global_step, _ = run_epoch(model, train_loader, criterion, optimizer, device, "train", {}, writer, global_step)
        val_loss, _, _ = run_epoch(model, val_loader, criterion, None, device, "val", {})

        if local_rank == 0:
            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Val", val_loss, epoch + 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                torch.save(model.module.state_dict(), best_model_path)
            else:
                counter += 1
                if counter >= patience:
                    break

            if (epoch + 1) % save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "counter": counter,
                    "global_step": global_step
                }, ckpt_path)

    model.module.load_state_dict(torch.load(config['save_dir'].format(iteration_idx=iteration_idx, mode=finetune_mode)))
    _, _, per_file_preds = run_epoch(model, test_loader, criterion, None, device, "test", test_metrics)
    test_results = compute_metrics(test_metrics)

    if local_rank == 0:
        for k, v in test_results.items():
            writer.add_scalar(f"Test/{k}", v)

        for file, results in per_file_preds.items():
            y_true = results["y_true"]
            y_pred = results["y_pred"]
            correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
            acc = correct / len(y_true) if y_true else 0.0
            print(f"{file:35s} | Accuracy: {acc:.4f}")
        writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    dataset_path = config["dataset_path"]
    all_patients = sorted([p for p in os.listdir(dataset_path) if not p.startswith(".")])[:6]
    splits = leave_one_out_splits(all_patients, val_count=2)

    for idx, split in enumerate(splits):
        train_loader = make_loader(split["train"], dataset_path, config, shuffle=True)
        val_loader = make_loader(split["val"], dataset_path, config, shuffle=False)
        test_loader = make_loader(split["test"], dataset_path, config, shuffle=False)
        supervised(config, train_loader, val_loader, test_loader, idx + 1)