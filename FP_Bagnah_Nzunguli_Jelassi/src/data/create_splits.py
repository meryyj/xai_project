# src/data/create_splits.py

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    csv_path: str | Path,
    out_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """
    Create stratified train/val/test splits from a single CSV listing all images.

    Input CSV format:
        image_path,label

    Parameters
    ----------
    csv_path : str | Path
        Path to the CSV file listing all images and labels.
    out_dir : str | Path
        Directory where train.csv, val.csv, and test.csv will be saved.
    train_ratio : float
        Proportion of data to use for training.
    val_ratio : float
        Proportion for validation.
    test_ratio : float
        Proportion for testing.
    seed : int
        Random seed for reproducibility.
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns or "label" not in df.columns:
        raise ValueError("Input CSV must contain 'image_path' and 'label' columns.")

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    X = df["image_path"]
    y = df["label"]

    # First: train vs temp (val+test)
    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=temp_ratio,
        stratify=y,
        random_state=seed,
    )

    # Then: split temp into val vs test
    val_relative = val_ratio / temp_ratio  # proportion of temp that goes to val

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1.0 - val_relative),
        stratify=y_temp,
        random_state=seed,
    )

    train_df = pd.DataFrame({"image_path": X_train, "label": y_train})
    val_df = pd.DataFrame({"image_path": X_val, "label": y_val})
    test_df = pd.DataFrame({"image_path": X_test, "label": y_test})

    train_out = out_dir / "train.csv"
    val_out = out_dir / "val.csv"
    test_out = out_dir / "test.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"Saved train split to: {train_out} ({len(train_df)} samples)")
    print(f"Saved val split   to: {val_out} ({len(val_df)} samples)")
    print(f"Saved test split  to: {test_out} ({len(test_df)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits for RAF-DB.")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV listing all images and labels.")
    parser.add_argument("--out_dir", type=str, default="data/splits", help="Output directory for split CSVs.")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio.")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio.")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()

    create_splits(
        csv_path=args.csv,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
