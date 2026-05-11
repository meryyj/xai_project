from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation.xai_common import make_baseline_tensor, normalize_map, predict_probabilities


def build_template_region_masks(height: int, width: int) -> Dict[str, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width]
    x_norm = (xx + 0.5) / width
    y_norm = (yy + 0.5) / height

    face = (
        ((x_norm - 0.5) / 0.34) ** 2
        + ((y_norm - 0.52) / 0.42) ** 2
    ) <= 1.0
    eyes = (
        (y_norm >= 0.22)
        & (y_norm <= 0.42)
        & (x_norm >= 0.18)
        & (x_norm <= 0.82)
    )
    mouth = (
        (y_norm >= 0.62)
        & (y_norm <= 0.84)
        & (x_norm >= 0.25)
        & (x_norm <= 0.75)
    )
    background = ~face

    return {
        "face": face,
        "eyes": eyes,
        "mouth": mouth,
        "background": background,
    }


def _occlude_region(
    image_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    mask: np.ndarray,
) -> torch.Tensor:
    pixel_mask = torch.from_numpy(mask.astype(np.float32)).to(image_tensor.device).unsqueeze(0)
    return image_tensor * (1.0 - pixel_mask) + baseline_tensor * pixel_mask


def analyze_masking(
    model: nn.Module,
    image_tensor: torch.Tensor,
    heatmap: np.ndarray,
    target_class: int,
    device: torch.device,
    *,
    regions: Sequence[str] = ("eyes", "mouth", "background"),
    baseline_strategy: str = "blur",
    blur_kernel_size: int = 11,
) -> list[dict[str, Any]]:
    image_tensor = image_tensor.to(device)
    base_probs = predict_probabilities(model, image_tensor, device).squeeze(0).numpy()
    base_pred = int(base_probs.argmax())
    base_conf = float(base_probs[target_class])
    baseline = make_baseline_tensor(
        image_tensor,
        strategy=baseline_strategy,
        blur_kernel_size=blur_kernel_size,
    )

    region_masks = build_template_region_masks(image_tensor.shape[1], image_tensor.shape[2])
    heatmap = normalize_map(heatmap)
    target_hw = tuple(image_tensor.shape[-2:])
    if heatmap.shape != target_hw:
        heatmap_tensor = torch.from_numpy(heatmap).float().view(1, 1, *heatmap.shape)
        heatmap = (
            F.interpolate(
                heatmap_tensor,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .numpy()
        )
        heatmap = normalize_map(heatmap)

    total_attr = float(heatmap.sum()) if float(heatmap.sum()) > 0 else 1.0

    rows = []
    for region_name in regions:
        if region_name not in region_masks:
            continue
        mask = region_masks[region_name]
        masked_tensor = _occlude_region(image_tensor, baseline, mask)
        probs = predict_probabilities(model, masked_tensor, device).squeeze(0).numpy()
        pred_after = int(probs.argmax())
        conf_after = float(probs[target_class])

        attribution_mass = float((heatmap * mask.astype(np.float32)).sum() / total_attr)
        rows.append(
            {
                "region": region_name,
                "masked_fraction": float(mask.mean()),
                "attribution_mass": attribution_mass,
                "confidence_after_masking": conf_after,
                "confidence_drop": base_conf - conf_after,
                "prediction_after_masking": pred_after,
                "prediction_changed": int(pred_after != base_pred),
            }
        )
    return rows
