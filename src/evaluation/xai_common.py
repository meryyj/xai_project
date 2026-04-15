from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.datasets import EMOTION_LABELS, RAFDBDataset
from src.evaluation.grad_cam import GradCAM, get_last_conv_layer, overlay_heatmap_on_image


@dataclass
class ExplanationResult:
    method: str
    heatmap: np.ndarray
    target_class: int
    predicted_class: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_map(heatmap: np.ndarray) -> np.ndarray:
    heatmap = np.nan_to_num(np.asarray(heatmap, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if heatmap.size == 0:
        return heatmap
    min_val = float(heatmap.min())
    max_val = float(heatmap.max())
    if max_val <= min_val:
        return np.zeros_like(heatmap, dtype=np.float32)
    return (heatmap - min_val) / (max_val - min_val)


def make_baseline_tensor(
    image_tensor: torch.Tensor,
    strategy: str = "blur",
    blur_kernel_size: int = 11,
) -> torch.Tensor:
    strategy = strategy.lower()
    if strategy == "zero":
        return torch.zeros_like(image_tensor)
    if strategy == "mean":
        mean_value = image_tensor.mean(dim=(1, 2), keepdim=True)
        return mean_value.expand_as(image_tensor)
    if strategy == "blur":
        kernel_size = max(3, int(blur_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2
        x = image_tensor.unsqueeze(0)
        x_padded = F.pad(x, (padding, padding, padding, padding), mode="reflect")
        blurred = F.avg_pool2d(x_padded, kernel_size=kernel_size, stride=1)
        return blurred.squeeze(0)
    raise ValueError(f"Unknown baseline strategy: {strategy}")


def predict_probabilities(
    model: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs.to(device))
        probs = torch.softmax(logits, dim=1)
    return probs.cpu()


def restore_model_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state")
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        raise KeyError(
            "Checkpoint must contain either 'model_state' or 'model_state_dict'."
        )
    model.load_state_dict(state_dict)
    return checkpoint


def collect_dataset_predictions(
    model: nn.Module,
    dataset: RAFDBDataset,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    all_probs = []
    all_labels = []
    for inputs, labels in loader:
        probs = predict_probabilities(model, inputs, device)
        all_probs.append(probs.numpy())
        all_labels.append(labels.numpy())

    y_probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = y_probs.argmax(axis=1)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probs": y_probs,
    }


def sample_indices_per_class(
    y_true: np.ndarray,
    samples_per_class: int,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[int]:
    rng = np.random.default_rng(seed)
    indices: list[int] = []
    for class_idx in sorted(np.unique(y_true).tolist()):
        class_indices = np.where(y_true == class_idx)[0]
        if len(class_indices) == 0:
            continue
        sample_count = min(samples_per_class, len(class_indices))
        chosen = rng.choice(class_indices, size=sample_count, replace=False)
        indices.extend(int(idx) for idx in chosen.tolist())

    indices = sorted(set(indices))
    if max_samples is not None and len(indices) > max_samples:
        indices = sorted(rng.choice(indices, size=max_samples, replace=False).tolist())
    return indices


def tensor_to_display_image(
    image_tensor: torch.Tensor,
    cfg: Dict[str, Any] | None = None,
) -> np.ndarray:
    image = image_tensor.detach().cpu().float()
    if cfg is not None:
        normalize_cfg = cfg.get("preprocessing", {}).get("normalize", {})
        mean = torch.tensor(normalize_cfg.get("mean", [0.0]), dtype=image.dtype).view(-1, 1, 1)
        std = torch.tensor(normalize_cfg.get("std", [1.0]), dtype=image.dtype).view(-1, 1, 1)
        if mean.shape[0] == image.shape[0]:
            image = image * std + mean

    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)

    image = image.clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    return image


def save_explanation_figure(
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    cfg: Dict[str, Any],
    output_path: str | Path,
    *,
    title: str,
    alpha: float = 0.45,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_display_image(image_tensor, cfg)
    overlay = overlay_heatmap_on_image((image * 255).astype(np.uint8), normalize_map(heatmap), alpha=alpha)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(normalize_map(heatmap), cmap="jet")
    axes[1].set_title("Heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    for ax in axes:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_method_comparison_figure(
    image_tensor: torch.Tensor,
    cfg: Dict[str, Any],
    results: list[ExplanationResult],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = tensor_to_display_image(image_tensor, cfg)
    n_cols = len(results) + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4.5))

    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[0].axis("off")

    for idx, result in enumerate(results, start=1):
        overlay = overlay_heatmap_on_image(
            (image * 255).astype(np.uint8),
            normalize_map(result.heatmap),
            alpha=0.45,
        )
        axes[idx].imshow(overlay)
        axes[idx].set_title(
            f"{result.method}\n{EMOTION_LABELS[result.predicted_class]} ({result.confidence:.2%})"
        )
        axes[idx].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class GradCAMExplainer:
    name = "gradcam"

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.target_layer = get_last_conv_layer(model)
        self.grad_cam = GradCAM(model, self.target_layer)

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> ExplanationResult:
        probs = predict_probabilities(self.model, image_tensor, self.device).squeeze(0).numpy()
        predicted_class = int(probs.argmax())
        if target_class is None:
            target_class = predicted_class
        confidence = float(probs[target_class])

        heatmap = self.grad_cam(image_tensor.unsqueeze(0).to(self.device), target_class=target_class)
        heatmap = normalize_map(heatmap)

        return ExplanationResult(
            method=self.name,
            heatmap=heatmap,
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            metadata={"target_layer": str(self.target_layer)},
        )
