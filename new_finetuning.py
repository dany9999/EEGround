import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from itertools import combinations
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.SupervisedClassifier import BIOTClassifier  # Assumiamo che prenda encoder + classifier
from model.SelfSupervisedPretrainEMB import UnsupervisedPretrain
from utils import CHBMITLoader, load_config, BinaryBalancedAccuracy
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


# ==== Data Loader ====

def make_loader(patients_list, config, shuffle=False):
    segment_files = []
    root = "CHB-MIT/clean_segments"
    for patient in patients_list:
        patient_path = os.path.join(root, patient)
        if os.path.exists(patient_path):
            files = [os.path.join(patient, f) for f in os.listdir(patient_path) if f.endswith(".pkl")]
            segment_files.extend(files)
    dataset = CHBMITLoader(root, segment_files, config["sampling_rate"])
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        drop_last=shuffle,  # drop_last only for train
        num_workers=config["num_workers"]
    )


# ==== Train / Eval ====

def run_epoch(model, dataloader, criterion, optimizer, device, mode, metrics, writer=None, global_step=0):
    if mode == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0

    # Reset metrics before epoch
    for metric in metrics.values():
        metric.reset()

    for x, y in tqdm(dataloader, desc=mode.capitalize()):
        x, y = x.to(device), y.to(device).float().view(-1, 1)

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

        if writer and mode == "train":
            writer.add_scalar("BatchLoss/Train", loss.item(), global_step)
            global_step += 1

    avg_loss = running_loss / len(dataloader)
    return avg_loss, global_step


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
    optimizer = optim.Adam(model.classifier.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
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

    patience = config["patience"]
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
    run_epoch(model, test_loader, criterion, None, device, "test", test_metrics)
    test_results = compute_metrics(test_metrics)

    print(f"\n=== Split {iteration_idx} Test Results ===")
    for k, v in test_results.items():
        print(f"{k.upper():7s}: {v:.4f}")
        writer.add_scalar(f"Test/{k}", v)

    writer.close()


# ==== Main ====

if __name__ == "__main__":
    config = load_config("configs/finetuning.yml")
    all_patients = sorted(os.listdir("CHB-MIT/Numpy"))
    
    splits = leave_one_out_splits(all_patients, val_count=2)

    for idx, split in enumerate(splits):
        print(f"\n--- Running Split {idx + 1}/{len(splits)} ---")
        train_loader = make_loader(split["train"], config, shuffle=True)
        val_loader = make_loader(split["val"], config, shuffle=False)
        test_loader = make_loader(split["test"], config, shuffle=False)
        supervised(config, train_loader, val_loader, test_loader, idx + 1)