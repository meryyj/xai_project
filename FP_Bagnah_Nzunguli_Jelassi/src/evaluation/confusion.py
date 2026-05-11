from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    from src.data.datasets import EMOTION_LABELS
except ModuleNotFoundError:
    # Keep this module usable even when the deep learning stack is not installed yet.
    EMOTION_LABELS = [
        "neutral",
        "happy",
        "sad",
        "surprise",
        "fear",
        "disgust",
        "anger",
    ]
from src.utils.logging import setup_logging

logger = logging.getLogger("confusion")


def compute_confusion_matrix(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    labels: Sequence[int] | None = None,
    normalize: str | None = None,
) -> np.ndarray:
    """
    Wrapper around sklearn's confusion_matrix with a stable ndarray return type.
    """
    if labels is None:
        labels = sorted(set(np.asarray(y_true).tolist()) | set(np.asarray(y_pred).tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    return np.asarray(cm)


def analyze_confusion_pairs(
    cm: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Return the most common off-diagonal confusions.
    """
    rows: list[dict[str, object]] = []
    for true_idx, true_name in enumerate(class_names):
        row_total = float(cm[true_idx].sum())
        for pred_idx, pred_name in enumerate(class_names):
            if true_idx == pred_idx:
                continue
            count = float(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            rows.append(
                {
                    "true_label": true_name,
                    "pred_label": pred_name,
                    "count": int(round(count)),
                    "row_error_rate": 0.0 if row_total == 0 else count / row_total,
                }
            )

    pairs_df = pd.DataFrame(rows)
    if pairs_df.empty:
        return pd.DataFrame(
            columns=["true_label", "pred_label", "count", "row_error_rate"]
        )

    return pairs_df.sort_values(
        by=["count", "row_error_rate"],
        ascending=[False, False],
    ).head(top_k)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    value_format: str,
    cmap: str = "Blues",
) -> None:
    """
    Save a confusion matrix heatmap with inline annotations.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = float(cm.max()) / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], value_format),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix figure to: {output_path}")


def _load_predictions(predictions_csv: str | Path) -> tuple[np.ndarray, np.ndarray]:
    predictions_csv = Path(predictions_csv)
    if not predictions_csv.is_file():
        raise FileNotFoundError(f"Predictions CSV not found: {predictions_csv}")

    pred_df = pd.read_csv(predictions_csv)
    required_columns = {"y_true", "y_pred"}
    missing = required_columns - set(pred_df.columns)
    if missing:
        raise ValueError(
            f"Predictions CSV must contain columns {sorted(required_columns)}, missing {sorted(missing)}."
        )

    return pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate confusion matrix artifacts")
    parser.add_argument(
        "--predictions_csv",
        type=str,
        required=True,
        help="Path to the predictions CSV produced by evaluation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/evaluation/confusion",
        help="Directory where confusion artifacts will be saved.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="test_confusion",
        help="Prefix for generated files.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="How many top confusion pairs to save.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir / "confusion.log")

    y_true, y_pred = _load_predictions(args.predictions_csv)
    labels = list(range(len(EMOTION_LABELS)))

    cm_counts = compute_confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    counts_df = pd.DataFrame(cm_counts, index=EMOTION_LABELS, columns=EMOTION_LABELS)
    norm_df = pd.DataFrame(cm_norm, index=EMOTION_LABELS, columns=EMOTION_LABELS)

    counts_csv = output_dir / f"{args.prefix}_counts.csv"
    norm_csv = output_dir / f"{args.prefix}_normalized_true.csv"
    counts_df.to_csv(counts_csv)
    norm_df.to_csv(norm_csv)
    logger.info(f"Saved confusion matrix counts to: {counts_csv}")
    logger.info(f"Saved normalized confusion matrix to: {norm_csv}")

    plot_confusion_matrix(
        cm_counts,
        EMOTION_LABELS,
        output_dir / f"{args.prefix}_counts.png",
        title="Confusion Matrix (Counts)",
        value_format="d",
    )
    plot_confusion_matrix(
        cm_norm,
        EMOTION_LABELS,
        output_dir / f"{args.prefix}_normalized_true.png",
        title="Confusion Matrix (Row-normalized)",
        value_format=".2f",
    )

    pairs_df = analyze_confusion_pairs(cm_counts, EMOTION_LABELS, top_k=args.top_k)
    pairs_csv = output_dir / f"{args.prefix}_top_confusions.csv"
    pairs_df.to_csv(pairs_csv, index=False)
    logger.info(f"Saved top confusion pairs to: {pairs_csv}")


if __name__ == "__main__":
    main()
