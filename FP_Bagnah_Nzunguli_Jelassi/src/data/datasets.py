# src/data/datasets.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from .transforms import build_transforms

# Standard 7 basic emotion labels (you can adjust names if your CSV differs)
EMOTION_LABELS: List[str] = [
    "neutral",
    "happy",
    "sad",
    "surprise",
    "fear",
    "disgust",
    "anger",
]

LABEL_TO_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(EMOTION_LABELS)}


@dataclass
class RAFDBSample:
    """
    Simple container for a single sample entry loaded from the CSV.
    """
    image_path: Path
    label_idx: int


class RAFDBDataset(Dataset):
    """
    Generic dataset for RAF-DB-style splits based on a CSV file.

    Expected CSV format:
        image_path,label

    - image_path: path to the image, relative to `root` directory
    - label: either string emotion name (e.g., 'happy') or integer class id

    The dataset will map string labels via LABEL_TO_INDEX if necessary.
    """

    def __init__(
        self,
        root: str | Path,
        split_csv: str | Path,
        transform=None,
        label_to_index: Optional[Dict[str, int]] = None,
    ):
        self.root = Path(root)
        self.split_csv = Path(split_csv)
        self.transform = transform

        if label_to_index is None:
            self.label_to_index = LABEL_TO_INDEX
        else:
            self.label_to_index = label_to_index

        if not self.split_csv.is_file():
            raise FileNotFoundError(f"Split file not found: {self.split_csv}")

        df = pd.read_csv(self.split_csv)

        if "image_path" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"CSV {self.split_csv} must have at least two columns: 'image_path' and 'label'"
            )

        self.samples: List[RAFDBSample] = []

        for _, row in df.iterrows():
            rel_path = Path(str(row["image_path"]))
            label_raw = row["label"]

            # Map label to index
            if isinstance(label_raw, str):
                label_key = label_raw.strip().lower()
                if label_key not in self.label_to_index:
                    raise ValueError(
                        f"Label '{label_raw}' (normalized '{label_key}') not found in label_to_index mapping."
                    )
                label_idx = self.label_to_index[label_key]
            else:
                # assume numeric label
                label_idx = int(label_raw)

            img_path = self.root / rel_path
            self.samples.append(RAFDBSample(image_path=img_path, label_idx=label_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]
        img = Image.open(sample.image_path).convert("RGB")  # convert to RGB first

        if self.transform is not None:
            img = self.transform(img)

        label = sample.label_idx
        return img, label


def build_dataloaders(cfg: Dict[str, Any]) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders using the config dictionary.

    Assumes:
      cfg["dataset"]["root"]       -> data root (e.g., "data/processed")
      cfg["dataset"]["train_split"] -> e.g., "data/splits/train.csv"
      cfg["dataset"]["val_split"]   -> ...
      cfg["dataset"]["test_split"]  -> ...

    and basic training params:
      cfg["training"]["batch_size"]
    """
    batch_size = cfg["training"]["batch_size"]
    num_workers = cfg["training"].get("num_workers", 4)
    pin_memory = cfg["training"].get("pin_memory", True)

    root = cfg["dataset"]["root"]
    splits_dir = cfg["dataset"]["splits_dir"]

    train_csv = Path(splits_dir) / cfg["dataset"]["train_split"]
    val_csv = Path(splits_dir) / cfg["dataset"]["val_split"]
    test_csv = Path(splits_dir) / cfg["dataset"]["test_split"]

    # Build transforms
    train_tf = build_transforms(cfg, split="train")
    val_tf = build_transforms(cfg, split="val")
    test_tf = build_transforms(cfg, split="test")

    # Build datasets
    train_ds = RAFDBDataset(root=root, split_csv=train_csv, transform=train_tf)
    val_ds = RAFDBDataset(root=root, split_csv=val_csv, transform=val_tf)
    test_ds = RAFDBDataset(root=root, split_csv=test_csv, transform=test_tf)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
