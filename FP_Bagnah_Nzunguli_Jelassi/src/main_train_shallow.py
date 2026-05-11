# src/main_train_shallow.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.datasets import build_dataloaders
from src.models.shallow_cnn import ShallowCNN


logger = logging.getLogger("train_shallow")


def compute_class_weights(train_loader, num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, labels in train_loader:
        labels = labels.view(-1)
        counts += torch.bincount(labels, minlength=num_classes)

    total = counts.sum().item()
    # Avoid division by zero
    counts = counts.clamp(min=1)
    weights = total / (num_classes * counts.float())
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_y, all_pred = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_pred.extend(preds)
        all_y.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    return avg_loss, acc, macro_f1


def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_y, all_pred = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size

            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_pred.extend(preds)
            all_y.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_y, all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    return avg_loss, acc, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Train shallow CNN baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_template.yaml",
        help="Path to YAML config (dataset & training params).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs (otherwise use config).",
    )
    args = parser.parse_args()

    # Logging
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    Path("experiments/checkpoints").mkdir(parents=True, exist_ok=True)
    setup_logging("experiments/logs/train_shallow.log")

    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)

    seed = cfg.data.get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataloaders
    loaders = build_dataloaders(cfg.data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    num_classes = 7  # RAF-DB 7 emotions
    in_channels = int(cfg.data["dataset"].get("channels", 1))

    # Model
    model = ShallowCNN(in_channels=in_channels, num_classes=num_classes).to(device)
    logger.info(f"ShallowCNN:\n{model}")

    # Class weights for imbalance
    logger.info("Computing class weights from training set...")
    class_weights = compute_class_weights(train_loader, num_classes=num_classes)
    logger.info(f"Class counts-derived weights: {class_weights.tolist()}")
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer – simple Adam for baseline
    base_lr = float(cfg.data.get("optimizer", {}).get("lr", 1e-3))
    weight_decay = float(cfg.data.get("optimizer", {}).get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
    )
    logger.info(f"Using Adam optimizer (lr={base_lr}, weight_decay={weight_decay})")

    num_epochs = args.epochs or int(cfg.data["training"].get("num_epochs", 50))
    patience = int(cfg.data["training"].get("early_stopping_patience", 10))
    logger.info(
        f"Starting training for {num_epochs} epochs (early stopping patience={patience})"
    )

    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_ckpt_path = Path("experiments/checkpoints/shallow_best.pt")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = eval_one_epoch(
            model, val_loader, criterion, device
        )

        logger.info(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}, macroF1={train_f1:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, macroF1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1 + 1e-4:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_macro_f1": best_val_f1,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
            logger.info(
                f"New best val_macro_f1={best_val_f1:.4f} at epoch {epoch}, "
                f"checkpoint saved to {best_ckpt_path}"
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {patience} epochs)."
                )
                break

    # Load best checkpoint and evaluate on test set
    if best_ckpt_path.is_file():
        logger.info(f"Loading best checkpoint from {best_ckpt_path} for TEST evaluation.")
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("No best checkpoint found, evaluating last epoch model on TEST.")

    test_loss, test_acc, test_f1 = eval_one_epoch(
        model, test_loader, criterion, device
    )
    logger.info(
        f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}, macroF1={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()
