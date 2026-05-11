# src/main_dummy_baseline.py

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logging import setup_logging
from src.data.datasets import build_dataloaders


logger = logging.getLogger("dummy_baseline")


def compute_majority_class(train_loader):
    """
    Iterate once over the training loader and find the majority class.
    Returns: (majority_label_idx, counts_dict)
    """
    counts = Counter()
    for _, labels in train_loader:
        labels = labels.detach().cpu().numpy().tolist()
        counts.update(labels)

    majority_label, majority_count = counts.most_common(1)[0]
    return majority_label, counts


def evaluate_constant(loader, constant_label: int):
    """
    Evaluate a constant classifier that always predicts 'constant_label'.
    Returns: (accuracy, macro_f1, y_true, y_pred)
    """
    all_y, all_pred = [], []
    for _, labels in loader:
        labels = labels.detach().cpu().numpy().tolist()
        all_y.extend(labels)
        all_pred.extend([constant_label] * len(labels))

    acc = accuracy_score(all_y, all_pred)
    macro_f1 = f1_score(all_y, all_pred, average="macro")
    return acc, macro_f1, np.array(all_y), np.array(all_pred)


def main():
    parser = argparse.ArgumentParser(description="Dummy baseline: majority class.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_template.yaml",
        help="Path to YAML config (for dataset/splits).",
    )
    args = parser.parse_args()

    # Logging
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    setup_logging("experiments/logs/dummy_baseline.log")

    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)

    seed = cfg.data.get("seed", 42)
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    # Build dataloaders with same pipeline as main model
    loaders = build_dataloaders(cfg.data)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    logger.info("Computing majority class from training set...")
    majority_label, counts = compute_majority_class(train_loader)
    logger.info(f"Class counts: {dict(counts)}")
    logger.info(f"Majority class index: {majority_label}")

    # Evaluate on all splits
    for split_name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        acc, macro_f1, y_true, y_pred = evaluate_constant(loader, majority_label)
        logger.info(
            f"[{split_name.upper()}] acc={acc:.4f}, macroF1={macro_f1:.4f}"
        )
        if split_name == "test":
            logger.info("Classification report on TEST set:\n" +
                        classification_report(y_true, y_pred, digits=4))

    logger.info("Dummy baseline evaluation complete.")


if __name__ == "__main__":
    main()
