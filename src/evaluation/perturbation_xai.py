from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from src.evaluation.xai_common import (
    ExplanationResult,
    make_baseline_tensor,
    normalize_map,
    predict_probabilities,
)


def create_grid_segments(height: int, width: int, rows: int = 5, cols: int = 5) -> np.ndarray:
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    segments = np.zeros((height, width), dtype=np.int32)
    y_edges = np.linspace(0, height, rows + 1, dtype=np.int32)
    x_edges = np.linspace(0, width, cols + 1, dtype=np.int32)

    seg_id = 0
    for row_idx in range(rows):
        for col_idx in range(cols):
            y0, y1 = y_edges[row_idx], y_edges[row_idx + 1]
            x0, x1 = x_edges[col_idx], x_edges[col_idx + 1]
            segments[y0:y1, x0:x1] = seg_id
            seg_id += 1
    return segments


def coalition_to_tensor_mask(
    coalition: np.ndarray,
    segments: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    mask = coalition[segments]
    return torch.from_numpy(mask.astype(np.float32)).to(device)


def apply_coalition(
    image_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    coalition: np.ndarray,
    segments: np.ndarray,
) -> torch.Tensor:
    mask = coalition_to_tensor_mask(coalition, segments, image_tensor.device).unsqueeze(0)
    return image_tensor * mask + baseline_tensor * (1.0 - mask)


def sample_random_coalitions(
    num_segments: int,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    coalitions = rng.integers(0, 2, size=(num_samples, num_segments), endpoint=False).astype(np.float32)
    anchors = [
        np.zeros((1, num_segments), dtype=np.float32),
        np.ones((1, num_segments), dtype=np.float32),
    ]
    return np.concatenate(anchors + [coalitions], axis=0)


def evaluate_coalitions(
    model: nn.Module,
    image_tensor: torch.Tensor,
    baseline_tensor: torch.Tensor,
    segments: np.ndarray,
    coalitions: np.ndarray,
    target_class: int,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    scores = []
    batch_tensors = []

    for coalition in coalitions:
        perturbed = apply_coalition(image_tensor, baseline_tensor, coalition, segments)
        batch_tensors.append(perturbed)
        if len(batch_tensors) >= batch_size:
            batch = torch.stack(batch_tensors, dim=0)
            probs = predict_probabilities(model, batch, device).numpy()
            scores.extend(probs[:, target_class].tolist())
            batch_tensors = []

    if batch_tensors:
        batch = torch.stack(batch_tensors, dim=0)
        probs = predict_probabilities(model, batch, device).numpy()
        scores.extend(probs[:, target_class].tolist())

    return np.asarray(scores, dtype=np.float32)


def lime_kernel_weights(coalitions: np.ndarray, kernel_width: float) -> np.ndarray:
    distances = np.sqrt(((1.0 - coalitions) ** 2).sum(axis=1))
    return np.exp(-(distances ** 2) / max(kernel_width ** 2, 1e-8))


def kernel_shap_weights(coalitions: np.ndarray) -> np.ndarray:
    num_segments = coalitions.shape[1]
    weights = np.zeros(coalitions.shape[0], dtype=np.float64)

    for idx, coalition in enumerate(coalitions):
        size = int(coalition.sum())
        if size == 0 or size == num_segments:
            weights[idx] = 1_000.0
            continue
        denom = math.comb(num_segments, size) * size * (num_segments - size)
        weights[idx] = (num_segments - 1) / max(float(denom), 1e-8)
    return weights.astype(np.float32)


def segment_scores_to_heatmap(
    segment_scores: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    return segment_scores[segments]


@dataclass
class PerturbationExplainerConfig:
    rows: int = 5
    cols: int = 5
    num_samples: int = 200
    batch_size: int = 32
    baseline_strategy: str = "blur"
    blur_kernel_size: int = 11
    ridge_alpha: float = 1.0
    kernel_width: float = 5.0
    random_seed: int = 42


class _BasePerturbationExplainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: PerturbationExplainerConfig,
    ):
        self.model = model
        self.device = device
        self.config = config

    def _fit_segment_scores(
        self,
        coalitions: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> ExplanationResult:
        image_tensor = image_tensor.to(self.device)
        probs = predict_probabilities(self.model, image_tensor, self.device).squeeze(0).numpy()
        predicted_class = int(probs.argmax())
        if target_class is None:
            target_class = predicted_class
        confidence = float(probs[target_class])

        _, height, width = image_tensor.shape
        segments = create_grid_segments(height, width, rows=self.config.rows, cols=self.config.cols)
        baseline = make_baseline_tensor(
            image_tensor,
            strategy=self.config.baseline_strategy,
            blur_kernel_size=self.config.blur_kernel_size,
        )

        rng = np.random.default_rng(self.config.random_seed)
        coalitions = sample_random_coalitions(
            num_segments=int(segments.max()) + 1,
            num_samples=self.config.num_samples,
            rng=rng,
        )
        scores = evaluate_coalitions(
            self.model,
            image_tensor,
            baseline,
            segments,
            coalitions,
            target_class,
            self.device,
            batch_size=self.config.batch_size,
        )
        segment_scores = self._fit_segment_scores(coalitions, scores)
        raw_heatmap = segment_scores_to_heatmap(segment_scores, segments)
        positive_heatmap = np.maximum(raw_heatmap, 0.0)
        if np.allclose(positive_heatmap.max(), 0.0):
            positive_heatmap = np.abs(raw_heatmap)
        heatmap = normalize_map(positive_heatmap)

        return ExplanationResult(
            method=self.name,
            heatmap=heatmap,
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            metadata={
                "segment_scores": segment_scores.tolist(),
                "rows": self.config.rows,
                "cols": self.config.cols,
                "num_samples": self.config.num_samples,
            },
        )


class LIMEExplainer(_BasePerturbationExplainer):
    name = "lime"

    def _fit_segment_scores(
        self,
        coalitions: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        weights = lime_kernel_weights(coalitions, kernel_width=self.config.kernel_width)
        regressor = Ridge(alpha=self.config.ridge_alpha, fit_intercept=True)
        regressor.fit(coalitions, scores, sample_weight=weights)
        return regressor.coef_.astype(np.float32)


class KernelSHAPExplainer(_BasePerturbationExplainer):
    name = "shap"

    def _fit_segment_scores(
        self,
        coalitions: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        weights = kernel_shap_weights(coalitions)
        regressor = Ridge(alpha=max(self.config.ridge_alpha, 1e-4), fit_intercept=True)
        regressor.fit(coalitions, scores, sample_weight=weights)
        return regressor.coef_.astype(np.float32)
