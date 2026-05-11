# src/main_classical_baseline.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import RAFDBDataset, EMOTION_LABELS

logger = logging.getLogger("classical_baseline")


def build_eval_transform(image_size: int = 32, grayscale: bool = True) -> T.Compose:
    """
    Simple, FAST transform for classical baseline:
    - optional grayscale
    - resize to small size (e.g. 32x32)
    - to tensor [0,1]
    No normalization: LogisticRegression will handle scaling.
    """
    transforms = []
    if grayscale:
        transforms.append(T.Grayscale(num_output_channels=1))
    transforms.append(T.Resize((image_size, image_size)))
    transforms.append(T.ToTensor())
    return T.Compose(transforms)


def build_loaders(
    cfg,
    batch_size: int = 128,
    image_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = cfg["dataset"]
    root = dataset_cfg["root"]
    splits_dir = dataset_cfg["splits_dir"]

    train_csv = Path(splits_dir) / dataset_cfg["train_split"]
    val_csv = Path(splits_dir) / dataset_cfg["val_split"]
    test_csv = Path(splits_dir) / dataset_cfg["test_split"]

    tf = build_eval_transform(image_size=image_size, grayscale=True)

    train_ds = RAFDBDataset(root=root, split_csv=train_csv, transform=tf)
    val_ds = RAFDBDataset(root=root, split_csv=val_csv, transform=tf)
    test_ds = RAFDBDataset(root=root, split_csv=test_csv, transform=tf)

    num_workers = cfg["training"].get("num_workers", 4)
    pin_memory = cfg["training"].get("pin_memory", True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,  # order doesn't matter, but deterministic is nice
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader


def collect_features(
    loader: DataLoader,
    max_samples: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten images and collect into NumPy arrays.
    Optionally limit to `max_samples` for speed.
    """
    xs = []
    ys = []

    for images, labels in loader:
        # images: (B, C, H, W)
        b = images.size(0)
        images = images.view(b, -1)  # flatten
        xs.append(images.cpu().numpy())
        ys.append(labels.cpu().numpy())

        if max_samples is not None and sum(len(y) for y in ys) >= max_samples:
            break

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)

    if max_samples is not None and X.shape[0] > max_samples:
        X = X[:max_samples]
        y = y[:max_samples]

    return X, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_template.yaml",
        help="Path to YAML config.",
    )
    # allow overriding max samples if wanted
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=2000)
    parser.add_argument("--max_test", type=int, default=2000)
    args = parser.parse_args()

    # Basic logging to console + file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    logger.info(f"Loading config from: {args.config}")
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    # Build loaders with small image size
    logger.info("Building dataloaders for classical baseline (32x32)...")
    train_loader, val_loader, test_loader = build_loaders(cfg, batch_size=256, image_size=32)

    logger.info("Collecting flattened features (this should be fairly quick)...")
    X_train, y_train = collect_features(train_loader, max_samples=args.max_train)
    X_val, y_val = collect_features(val_loader, max_samples=args.max_val)
    X_test, y_test = collect_features(test_loader, max_samples=args.max_test)

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logger.info(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    logger.info(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    # Simple scaling: divide by 1 (already in [0,1]), but you could add StandardScaler here if needed.
    # Train a multinomial Logistic Regression baseline
    logger.info("Fitting LogisticRegression baseline (multinomial, saga)...")
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        max_iter=200,
        n_jobs=-1,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    def eval_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
        y_pred = clf.predict(X)
        acc = accuracy_score(y, y_pred)
        macro_f1 = f1_score(y, y_pred, average="macro")
        logger.info(f"[{name}] Accuracy: {acc:.4f}, Macro-F1: {macro_f1:.4f}")
        print(f"\n=== {name} classification report ===")
        print(
            classification_report(
                y,
                y_pred,
                target_names=EMOTION_LABELS,
                digits=4,
            )
        )
        cm = confusion_matrix(y, y_pred)
        print(f"Confusion matrix ({name}):\n{cm}")

    eval_split("Train", X_train, y_train)
    eval_split("Val", X_val, y_val)
    eval_split("Test", X_test, y_test)


if __name__ == "__main__":
    main()
