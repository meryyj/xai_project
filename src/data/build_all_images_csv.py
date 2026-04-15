# src/data/build_all_images_csv.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from .datasets import EMOTION_LABELS


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def scan_images(images_root: str | Path) -> pd.DataFrame:
    """
    Scan a directory organized as:

        images_root/
          happy/
            img1.jpg
            ...
          sad/
            ...
          ...

    and build a DataFrame with columns:
        image_path (relative to images_root), label (emotion name, lowercase)

    Only files with VALID_EXTENSIONS are included.
    """
    images_root = Path(images_root)
    if not images_root.is_dir():
        raise NotADirectoryError(f"Images root does not exist or is not a directory: {images_root}")

    rows = []

    for class_dir in sorted(images_root.iterdir()):
        if not class_dir.is_dir():
            continue

        label = class_dir.name.strip().lower()
        if label not in EMOTION_LABELS:
            # You can choose to raise an error here instead if you want to enforce strict labels
            print(
                f"[WARN] Directory '{class_dir.name}' is not in EMOTION_LABELS "
                f"{EMOTION_LABELS}, skipping."
            )
            continue

        for img_path in class_dir.rglob("*"):
            if not img_path.is_file():
                continue
            if img_path.suffix.lower() not in VALID_EXTENSIONS:
                continue

            rel_path = img_path.relative_to(images_root)
            rows.append(
                {
                    "image_path": str(rel_path).replace("\\", "/"),  # make it OS-agnostic
                    "label": label,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No images found under {images_root} with extensions {VALID_EXTENSIONS}."
        )

    df = pd.DataFrame(rows)
    return df


def build_all_images_csv(images_root: str | Path, out_csv: str | Path) -> None:
    """
    Main helper: scan images_root and save a CSV listing all images and labels.

    Parameters
    ----------
    images_root : str | Path
        Root directory containing one subfolder per emotion.
    out_csv : str | Path
        Path where all_images.csv will be written.
    """
    images_root = Path(images_root)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = scan_images(images_root)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved all images listing to: {out_csv} (n={len(df)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Build all_images.csv from a directory structure label/image files."
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default="data/processed",
        help="Root directory with subfolders per class (e.g. data/processed/happy, ...).",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/all_images.csv",
        help="Output CSV path (default: data/all_images.csv).",
    )

    args = parser.parse_args()
    build_all_images_csv(args.images_root, args.out_csv)


if __name__ == "__main__":
    main()
