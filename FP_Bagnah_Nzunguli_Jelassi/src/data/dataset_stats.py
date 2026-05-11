# src/data/dataset_stats.py

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.data.datasets import EMOTION_LABELS


def compute_class_stats(split_csv: Path, label_col: str = "label"):
    """
    Read a split CSV and compute:
      - class counts
      - inverse-frequency class weights

    Works whether labels are strings (e.g. 'happy')
    or integer indices (0..6).
    """
    if not split_csv.is_file():
        raise FileNotFoundError(f"CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)

    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in {split_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    labels = df[label_col].tolist()
    counts = Counter(labels)
    total = sum(counts.values())
    n_classes = len(counts)

    print(f"\n=== Class counts in {split_csv.name} ===")
    for cls, cnt in sorted(counts.items(), key=lambda x: str(x[0])):
        pct = cnt / total if total > 0 else 0.0
        print(f"{str(cls):>10}: {cnt:5d}  ({pct:6.2%})")

    # Inverse frequency weights:
    #   w_i = N / (K * n_i)
    print("\n=== Suggested class weights (inverse frequency) ===")

    # If labels are strings like 'happy', keep that.
    # If labels are ints, map them back to EMOTION_LABELS where possible.
    weights = {}

    for cls, cnt in sorted(counts.items(), key=lambda x: str(x[0])):
        if cnt == 0:
            w = 0.0
        else:
            w = total / (n_classes * cnt)
        weights[cls] = w

    for cls, w in weights.items():
        print(f"{str(cls):>10}: {w:.4f}")

    # If labels are ints 0..6, also print them in EMOTION_LABELS order
    if all(isinstance(k, (int, float)) for k in counts.keys()):
        print("\n=== Weights in EMOTION_LABELS order ===")
        ordered = []
        for idx, name in enumerate(EMOTION_LABELS):
            if idx in weights:
                ordered.append(weights[idx])
            else:
                ordered.append(0.0)
            print(f"{idx} ({name:>8}): {ordered[-1]:.4f}")
        print("\nCopy this into your config if you like:")
        print(ordered)

    return counts, weights


def main():
    parser = argparse.ArgumentParser(
        description="Compute class distribution and suggested class weights."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_template.yaml",
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Which split CSV to analyze.",
    )
    parser.add_argument(
        "--label_col",
        type=str,
        default="label",
        help="Name of the label column in the CSV.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds_cfg = cfg["dataset"]

    splits_dir = Path(ds_cfg["splits_dir"])
    split_name = ds_cfg.get(f"{args.split}_split", f"{args.split}.csv")
    split_csv = splits_dir / split_name

    print(f"Using config: {args.config}")
    print(f"Analyzing split: {args.split} → {split_csv}")

    compute_class_stats(split_csv, label_col=args.label_col)


if __name__ == "__main__":
    main()
