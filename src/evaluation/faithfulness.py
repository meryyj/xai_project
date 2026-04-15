from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.evaluation.xai_common import make_baseline_tensor, normalize_map, predict_probabilities


@dataclass
class FaithfulnessResult:
    fractions: list[float]
    deletion_curve: list[float]
    insertion_curve: list[float]
    deletion_auc: float
    insertion_auc: float


def _build_topk_mask(
    importance_map: np.ndarray,
    fraction: float,
) -> np.ndarray:
    flat = importance_map.reshape(-1)
    num_pixels = flat.size
    num_keep = int(round(float(fraction) * num_pixels))
    if num_keep <= 0:
        return np.zeros_like(importance_map, dtype=bool)
    if num_keep >= num_pixels:
        return np.ones_like(importance_map, dtype=bool)

    topk_idx = np.argpartition(flat, -num_keep)[-num_keep:]
    mask = np.zeros(num_pixels, dtype=bool)
    mask[topk_idx] = True
    return mask.reshape(importance_map.shape)


def _apply_pixel_mask(
    image_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    mask: np.ndarray,
    mode: str,
) -> torch.Tensor:
    pixel_mask = torch.from_numpy(mask.astype(np.float32)).to(image_tensor.device).unsqueeze(0)
    if mode == "deletion":
        return image_tensor * (1.0 - pixel_mask) + baseline_tensor * pixel_mask
    if mode == "insertion":
        return baseline_tensor * (1.0 - pixel_mask) + image_tensor * pixel_mask
    raise ValueError(f"Unknown mode: {mode}")


def compute_faithfulness(
    model: nn.Module,
    image_tensor: torch.Tensor,
    importance_map: np.ndarray,
    target_class: int,
    device: torch.device,
    *,
    steps: int = 10,
    baseline_strategy: str = "blur",
    blur_kernel_size: int = 11,
) -> FaithfulnessResult:
    image_tensor = image_tensor.to(device)
    baseline_tensor = make_baseline_tensor(
        image_tensor,
        strategy=baseline_strategy,
        blur_kernel_size=blur_kernel_size,
    )

    importance_map = normalize_map(importance_map)
    fractions = np.linspace(0.0, 1.0, steps + 1)
    deletion_batch = []
    insertion_batch = []

    for fraction in fractions:
        mask = _build_topk_mask(importance_map, fraction=fraction)
        deletion_batch.append(_apply_pixel_mask(image_tensor, baseline_tensor, mask, mode="deletion"))
        insertion_batch.append(_apply_pixel_mask(image_tensor, baseline_tensor, mask, mode="insertion"))

    deletion_probs = predict_probabilities(model, torch.stack(deletion_batch, dim=0), device).numpy()
    insertion_probs = predict_probabilities(model, torch.stack(insertion_batch, dim=0), device).numpy()

    deletion_curve = deletion_probs[:, target_class].astype(np.float32)
    insertion_curve = insertion_probs[:, target_class].astype(np.float32)

    return FaithfulnessResult(
        fractions=fractions.astype(np.float32).tolist(),
        deletion_curve=deletion_curve.tolist(),
        insertion_curve=insertion_curve.tolist(),
        deletion_auc=float(np.trapz(deletion_curve, fractions)),
        insertion_auc=float(np.trapz(insertion_curve, fractions)),
    )


def plot_faithfulness_curves(
    result: FaithfulnessResult,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(result.fractions, result.deletion_curve, marker="o", label="Deletion")
    ax.plot(result.fractions, result.insertion_curve, marker="o", label="Insertion")
    ax.set_xlabel("Fraction of salient pixels modified")
    ax.set_ylabel("Target-class probability")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
