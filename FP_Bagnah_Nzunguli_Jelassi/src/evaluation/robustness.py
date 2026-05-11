from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.evaluation.xai_common import ExplanationResult, normalize_map


@dataclass
class RobustnessResult:
    perturbation_rows: list[dict[str, Any]]
    pearson_mean: float
    pearson_std: float
    rank_corr_mean: float
    rank_corr_std: float
    topk_iou_mean: float
    topk_iou_std: float


def _shift_tensor_with_padding(
    image_tensor: torch.Tensor,
    shift_y: int,
    shift_x: int,
) -> torch.Tensor:
    shifted = torch.zeros_like(image_tensor)

    src_y0 = max(0, -shift_y)
    src_y1 = image_tensor.shape[1] - max(0, shift_y)
    dst_y0 = max(0, shift_y)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    src_x0 = max(0, -shift_x)
    src_x1 = image_tensor.shape[2] - max(0, shift_x)
    dst_x0 = max(0, shift_x)
    dst_x1 = dst_x0 + (src_x1 - src_x0)

    shifted[:, dst_y0:dst_y1, dst_x0:dst_x1] = image_tensor[:, src_y0:src_y1, src_x0:src_x1]
    return shifted


def _perturb_image(
    image_tensor: torch.Tensor,
    rng: np.random.Generator,
    noise_std: float,
    brightness_delta: float,
    translation_px: int,
) -> torch.Tensor:
    perturbed = image_tensor.clone()
    if noise_std > 0:
        perturbed = perturbed + torch.randn_like(perturbed) * noise_std
    if brightness_delta > 0:
        delta = float(rng.uniform(-brightness_delta, brightness_delta))
        perturbed = perturbed + delta
    if translation_px > 0:
        shift_y = int(rng.integers(-translation_px, translation_px + 1))
        shift_x = int(rng.integers(-translation_px, translation_px + 1))
        perturbed = _shift_tensor_with_padding(perturbed, shift_y=shift_y, shift_x=shift_x)
    return perturbed


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    a_ranks = np.argsort(np.argsort(a_flat))
    b_ranks = np.argsort(np.argsort(b_flat))
    return _pearson_corr(a_ranks.astype(np.float32), b_ranks.astype(np.float32))


def _topk_iou(a: np.ndarray, b: np.ndarray, ratio: float = 0.2) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    k = max(1, int(round(ratio * a_flat.size)))

    a_idx = np.argpartition(a_flat, -k)[-k:]
    b_idx = np.argpartition(b_flat, -k)[-k:]

    a_mask = np.zeros_like(a_flat, dtype=bool)
    b_mask = np.zeros_like(b_flat, dtype=bool)
    a_mask[a_idx] = True
    b_mask[b_idx] = True

    intersection = np.logical_and(a_mask, b_mask).sum()
    union = np.logical_or(a_mask, b_mask).sum()
    return float(intersection / max(union, 1))


def evaluate_robustness(
    explainer,
    image_tensor: torch.Tensor,
    base_result: ExplanationResult,
    target_class: int,
    *,
    num_perturbations: int = 5,
    noise_std: float = 0.02,
    brightness_delta: float = 0.03,
    translation_px: int = 2,
    topk_ratio: float = 0.2,
    random_seed: int = 42,
) -> RobustnessResult:
    rng = np.random.default_rng(random_seed)
    base_map = normalize_map(base_result.heatmap)

    rows = []
    pearsons = []
    rank_corrs = []
    topk_ious = []

    for perturbation_idx in range(num_perturbations):
        perturbed = _perturb_image(
            image_tensor,
            rng=rng,
            noise_std=noise_std,
            brightness_delta=brightness_delta,
            translation_px=translation_px,
        )
        result = explainer.explain(perturbed, target_class=target_class)
        perturbed_map = normalize_map(result.heatmap)

        pearson = _pearson_corr(base_map, perturbed_map)
        rank_corr = _rank_corr(base_map, perturbed_map)
        topk_iou = _topk_iou(base_map, perturbed_map, ratio=topk_ratio)

        pearsons.append(pearson)
        rank_corrs.append(rank_corr)
        topk_ious.append(topk_iou)
        rows.append(
            {
                "perturbation_idx": perturbation_idx,
                "pearson": pearson,
                "rank_corr": rank_corr,
                "topk_iou": topk_iou,
                "target_confidence": result.confidence,
            }
        )

    return RobustnessResult(
        perturbation_rows=rows,
        pearson_mean=float(np.mean(pearsons)),
        pearson_std=float(np.std(pearsons)),
        rank_corr_mean=float(np.mean(rank_corrs)),
        rank_corr_std=float(np.std(rank_corrs)),
        topk_iou_mean=float(np.mean(topk_ious)),
        topk_iou_std=float(np.std(topk_ious)),
    )
