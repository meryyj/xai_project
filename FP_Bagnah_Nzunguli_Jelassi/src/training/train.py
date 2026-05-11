# src/training/train.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List

import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

from src.data.datasets import build_dataloaders
from src.models.factory import build_model_from_config
from src.training.explainability import (
    compute_explainability_loss,
    explainability_enabled,
)
from src.utils.config import Config

logger = logging.getLogger("train")


def _compute_class_counts(train_loader: DataLoader) -> List[int]:
    """
    Count how many samples per class are present in the training set.
    """
    logger.info("Computing class counts for training set...")
    class_counts: Dict[int, int] = {}
    for _, labels in train_loader:
        labels = labels.cpu().numpy()
        for y in labels:
            class_counts[int(y)] = class_counts.get(int(y), 0) + 1

    # Convert to sorted list by class index 0..K-1
    max_class = max(class_counts.keys())
    counts_list = [class_counts.get(i, 0) for i in range(max_class + 1)]
    logger.info(f"Class counts: {counts_list}")
    return counts_list


def _build_class_weights(class_counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Build inverse-frequency class weights:
        w_i = N / (K * n_i)
    """
    counts = np.array(class_counts, dtype=np.float32)
    N = counts.sum()
    K = len(counts)
    # Avoid division by zero for extremely rare / missing classes
    counts[counts == 0] = 1.0
    weights = N / (K * counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    logger.info(f"Class weights (inverse frequency): {weights.tolist()}")
    return weights_tensor


def _build_optimizer(model: nn.Module, optim_cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = optim_cfg.get("name", "adamw").lower()
    lr = float(optim_cfg.get("lr", 1e-3))
    weight_decay = float(optim_cfg.get("weight_decay", 1e-4))
    momentum = float(optim_cfg.get("momentum", 0.9))

    if name == "adamw":
        logger.info(f"Using AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        logger.info(f"Using Adam optimizer (lr={lr}, weight_decay={weight_decay})")
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        logger.info(
            f"Using SGD optimizer (lr={lr}, momentum={momentum}, weight_decay={weight_decay})"
        )
        return torch.optim.SGD(
            model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer name: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer, sched_cfg: Dict[str, Any]
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = sched_cfg.get("name", "none")
    name = name.lower() if isinstance(name, str) else "none"

    if name in ("none", "null", "", None):
        logger.info("No LR scheduler will be used.")
        return None

    if name == "cosine_annealing":
        T_max = int(sched_cfg.get("T_max", 50))
        eta_min = float(sched_cfg.get("eta_min", 1e-5))
        logger.info(
            f"Using CosineAnnealingLR scheduler (T_max={T_max}, eta_min={eta_min})"
        )
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )

    # Add other schedulers here if you want (ReduceLROnPlateau, OneCycleLR, etc.)
    logger.warning(f"Unknown scheduler name: {name}. No scheduler will be used.")
    return None


def _step_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool = True,
    explainability_cfg: Dict[str, Any] | None = None,
) -> Tuple[float, float, float, float]:
    """
    Run one epoch on the given loader.

    Returns:
      (avg_loss, accuracy, macro_f1, avg_explainability_loss)
    """
    explainability_cfg = explainability_cfg or {}
    use_explainability = train and explainability_enabled(explainability_cfg)

    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_explainability_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_explainability:
            inputs.requires_grad_(True)

        if train:
            optimizer.zero_grad()  # type: ignore

        with torch.set_grad_enabled(train):
            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)
            explainability_loss = None
            loss = classification_loss

            if use_explainability:
                explainability_loss = compute_explainability_loss(
                    inputs,
                    outputs,
                    labels,
                    explainability_cfg,
                )
                explainability_lambda = float(explainability_cfg.get("lambda", 0.0))
                loss = classification_loss + explainability_lambda * explainability_loss

            if train:
                loss.backward()
                optimizer.step()  # type: ignore

        running_loss += loss.item() * inputs.size(0)
        if explainability_loss is not None:
            running_explainability_loss += explainability_loss.item() * inputs.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    dataset_size = len(loader.dataset)
    avg_loss = running_loss / max(dataset_size, 1)

    all_preds_arr = np.array(all_preds)
    all_labels_arr = np.array(all_labels)

    accuracy = (all_preds_arr == all_labels_arr).mean().item()
    macro_f1 = f1_score(all_labels_arr, all_preds_arr, average="macro")
    avg_explainability_loss = running_explainability_loss / max(dataset_size, 1)

    return avg_loss, accuracy, macro_f1, avg_explainability_loss


def train_model(cfg: Config) -> None:
    """
    Main training entry point. Expects a Config object.
    """
    cfg_dict = cfg.data  # plain dict
    dataset_cfg = cfg_dict.get("dataset", {})
    training_cfg = cfg_dict.get("training", {})
    optim_cfg = cfg_dict.get("optimizer", {})
    sched_cfg = cfg_dict.get("scheduler", {})
    logging_cfg = cfg_dict.get("logging", {})
    explainability_cfg = cfg_dict.get("explainability_regularization", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Data loaders
    # ------------------------------------------------------------------
    loaders = build_dataloaders(cfg_dict)
    train_loader = loaders["train"]
    val_loader = loaders["val"]

    # ------------------------------------------------------------------
    # 2. Class weights (for imbalance)
    # ------------------------------------------------------------------
    class_counts = _compute_class_counts(train_loader)
    class_weights = _build_class_weights(class_counts, device=device)

    # ------------------------------------------------------------------
    # 3. Model, loss, optimizer, scheduler
    # ------------------------------------------------------------------
    model = build_model_from_config(cfg_dict).to(device)
    logger.info(f"Model built:\n{model}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = _build_optimizer(model, optim_cfg)
    scheduler = _build_scheduler(optimizer, sched_cfg)

    # ------------------------------------------------------------------
    # 4. Logging / checkpoints
    # ------------------------------------------------------------------
    log_dir = Path(logging_cfg.get("log_dir", "experiments/logs"))
    ckpt_dir = Path(logging_cfg.get("checkpoint_dir", "experiments/checkpoints"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    use_tb = bool(logging_cfg.get("tensorboard", True))
    writer: SummaryWriter | None = None
    if use_tb:
        writer = SummaryWriter(log_dir=str(log_dir / "tb"))

    num_epochs = int(training_cfg.get("num_epochs", 100))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 15))

    best_val_metric = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    save_best_metric_name = logging_cfg.get("save_best_metric", "val_macro_f1")

    logger.info(
        f"Starting training for {num_epochs} epochs. "
        f"Early stopping patience = {early_stopping_patience}, "
        f"best metric = {save_best_metric_name}"
    )
    if explainability_enabled(explainability_cfg):
        logger.info(
            "Explanation-guided regularization enabled with "
            f"lambda={float(explainability_cfg.get('lambda', 0.0)):.4f}"
        )

    # ------------------------------------------------------------------
    # 5. Epoch loop
    # ------------------------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc, train_macro_f1, train_explain_loss = _step_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            explainability_cfg=explainability_cfg,
        )

        # Validate
        val_loss, val_acc, val_macro_f1, _ = _step_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        # Step LR scheduler (if not ReduceLROnPlateau-style)
        if scheduler is not None:
            # For simplicity, we assume schedulers that step every epoch
            scheduler.step()

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch:03d}/{num_epochs:03d} "
            f"[{epoch_time:.1f}s] "
            f"Train: loss={train_loss:.4f}, acc={train_acc:.4f}, macroF1={train_macro_f1:.4f} | "
            f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}, macroF1={val_macro_f1:.4f}"
        )
        if explainability_enabled(explainability_cfg):
            logger.info(f"Epoch {epoch:03d} explainability_loss={train_explain_loss:.4f}")

        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("MacroF1/train", train_macro_f1, epoch)
            writer.add_scalar("MacroF1/val", val_macro_f1, epoch)
            if explainability_enabled(explainability_cfg):
                writer.add_scalar("Explainability/train", train_explain_loss, epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("LR", current_lr, epoch)

        # ------------------------------------------------------------------
        # 6. Check best metric & early stopping
        # ------------------------------------------------------------------
        if save_best_metric_name == "val_macro_f1":
            current_metric = val_macro_f1
        elif save_best_metric_name == "val_acc":
            current_metric = val_acc
        elif save_best_metric_name == "val_loss":
            current_metric = -val_loss  # because lower is better
        else:
            logger.warning(
                f"Unknown save_best_metric: {save_best_metric_name}, "
                f"defaulting to val_macro_f1."
            )
            current_metric = val_macro_f1

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_epoch = epoch
            epochs_no_improve = 0

            ckpt_path = ckpt_dir / "best_model.pt"
            logger.info(
                f"New best model at epoch {epoch} "
                f"({save_best_metric_name}={current_metric:.4f}). "
                f"Saving checkpoint to {ckpt_path}"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_metric": best_val_metric,
                    "class_counts": class_counts,
                    "config": cfg_dict,
                    "explainability_regularization": explainability_cfg,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1
            logger.info(
                f"No improvement in best metric for {epochs_no_improve} epoch(s). "
                f"Best so far: {best_val_metric:.4f} at epoch {best_epoch}"
            )

        if epochs_no_improve >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs. "
                f"Best epoch: {best_epoch} with {save_best_metric_name}={best_val_metric:.4f}"
            )
            break

    if writer is not None:
        writer.close()
