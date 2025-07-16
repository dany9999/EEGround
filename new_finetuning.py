import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import combinations
from torch.utils.tensorboard import SummaryWriter
from CHBMITLoader import  make_loader
from model.SupervisedClassifier import BIOTClassifier  # Assumiamo che prenda encoder + classifier
from utils import  load_config, BinaryBalancedAccuracy
from torchmetrics.classification import (
    BinaryAccuracy, BinaryAveragePrecision, BinaryAUROC, BinaryCohenKappa
)


# ==== Leave-One-Out Split ====

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




# ==== Train / Eval ====

def run_epoch(model, dataloader, criterion, optimizer, device, mode, metrics, writer=None, global_step=0):
    if mode == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    for metric in metrics.values():
        metric.reset()

    # Per salvare predizioni per file
    per_file_preds = {}

    for batch in tqdm(dataloader, desc=mode.capitalize()):
        if isinstance(batch, dict):  # Se il loader restituisce dizionario
            x = batch["x"].to(device)
            y = batch["y"].to(device).float().view(-1, 1)
            file_ids = batch["file"]
        else:
            x, y = batch
            x, y = x.to(device), y.to(device).float().view(-1, 1)
            file_ids = ["unknown_file"] * len(x)

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

        # Salva predizioni per file
        if mode == "test":
            for f_id, p, t in zip(file_ids, probs.detach().cpu(), y_int.cpu()):
                if f_id not in per_file_preds:
                    per_file_preds[f_id] = {"y_true": [], "y_pred": []}
                per_file_preds[f_id]["y_true"].append(t.item())
                per_file_preds[f_id]["y_pred"].append(int(p >= 0.5))

        if writer and mode == "train":
            writer.add_scalar("BatchLoss/Train", loss.item(), global_step)
            global_step += 1

    avg_loss = running_loss / len(dataloader)
    return avg_loss, global_step, per_file_preds if mode == "test" else None


def compute_metrics(metrics):
    results = {name: metric.compute().item() for name, metric in metrics.items()}
    for metric in metrics.values():
        metric.reset()
    return results


# ==== Supervised training ====

def supervised(config, train_loader, val_loader, test_loader, iteration_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Initialize model ===
    model = BIOTClassifier(
        emb_size=config['emb_size'],
        heads=config['heads'],
        depth=config['depth'],
        n_classes=config['n_classes']
    ).to(device)

    # === Load pretrained encoder weights ===
    pretrained_ckpt = config["pretrained_ckpt"]
    checkpoint = torch.load(pretrained_ckpt, map_location=device)
    # Load encoder weights strictly or loosely depending on checkpoint content
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    # === Setup optimizer: only classifier params are trainable ===
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))
    criterion = nn.BCEWithLogitsLoss()

    # === Setup metrics ===
    val_metrics = {
        "acc": BinaryAccuracy().to(device),
        "prauc": BinaryAveragePrecision().to(device),
        "auroc": BinaryAUROC().to(device),
        "balacc": BinaryBalancedAccuracy().to(device),
        "kappa": BinaryCohenKappa().to(device)
    }
    test_metrics = {k: v.clone() for k, v in val_metrics.items()}

    writer = SummaryWriter(log_dir=f"log-finetuning/run-{iteration_idx}")
    best_val_loss = float("inf")
    global_step = 0

    patience = config["early_stopping_patience"]
    counter = 0

    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        train_loss, global_step = run_epoch(
            model, train_loader, criterion, optimizer, device, "train", {}, writer, global_step
        )
        val_loss, _ = run_epoch(
            model, val_loader, criterion, None, device, "val", val_metrics
        )

        val_results = compute_metrics(val_metrics)

        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("Loss/Val", val_loss, epoch + 1)
        for k, v in val_results.items():
            writer.add_scalar(f"Val/{k}", v, epoch + 1)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_results['acc']:.4f}")

        # Early stopping + save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            save_path = config['save_path'].format(iteration_idx=iteration_idx)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # === Test ===
    model.load_state_dict(torch.load(config['save_path'].format(iteration_idx=iteration_idx)))
    _, _, per_file_preds = run_epoch(model, test_loader, criterion, None, device, "test", test_metrics)
    test_results = compute_metrics(test_metrics)

    print(f"\n=== Split {iteration_idx} Test Results ===")
    for k, v in test_results.items():
        print(f"{k.upper():7s}: {v:.4f}")
        writer.add_scalar(f"Test/{k}", v)

    # === Accuracy per file ===
    print("\n--- Accuracy per file ---")
    for file, results in per_file_preds.items():
        y_true = results["y_true"]
        y_pred = results["y_pred"]
        correct = sum(yt == yp for yt, yp in zip(y_true, y_pred))
        acc = correct / len(y_true) if y_true else 0.0
        print(f"{file:35s} | Accuracy: {acc:.4f}")


    writer.close()


# ==== Main ====

if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    all_patients = sorted([p for p in os.listdir("../../Datasets/chb-mit/data") if not p.startswith(".")])

    splits = leave_one_out_splits(all_patients, val_count=2)

    for idx, split in enumerate(splits):
        print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")
        train_loader = make_loader(split["train"], config, shuffle=True)
        val_loader = make_loader(split["val"], config, shuffle=False)
        test_loader = make_loader(split["test"], config, shuffle=False)
        supervised(config, train_loader, val_loader, test_loader, idx + 1)

