from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.evaluation.xai_common import ExplanationResult, normalize_map, predict_probabilities


class _TargetProbabilityWrapper(nn.Module):
    def __init__(self, model: nn.Module, target_class: int):
        super().__init__()
        self.model = model
        self.target_class = int(target_class)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.model(inputs)
        probs = torch.softmax(logits, dim=1)
        return probs[:, self.target_class : self.target_class + 1]


def _disable_inplace_modules(module: nn.Module) -> None:
    for child in module.modules():
        if hasattr(child, "inplace"):
            try:
                child.inplace = False
            except Exception:
                continue


def _build_explanation_model(model: nn.Module, device: torch.device) -> nn.Module:
    model_copy = copy.deepcopy(model).to(device)
    model_copy.eval()
    _disable_inplace_modules(model_copy)
    return model_copy


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _expected_value_to_float(expected_value: Any) -> float | None:
    if expected_value is None:
        return None
    array = _to_numpy(expected_value).reshape(-1)
    if array.size == 0:
        return None
    return float(array[0])


def _shap_values_to_heatmap(
    shap_values: Any,
    num_channels: int,
) -> np.ndarray:
    values = shap_values
    if isinstance(values, tuple):
        values = values[0]
    if isinstance(values, list):
        if len(values) == 0:
            raise ValueError("DeepSHAP returned an empty list of SHAP values.")
        values = values[0]

    values = _to_numpy(values)
    if values.ndim >= 5 and values.shape[-1] == 1:
        values = np.squeeze(values, axis=-1)

    if values.ndim == 4:
        values = values[0]

    if values.ndim == 3:
        if values.shape[0] == num_channels:
            raw_heatmap = values.sum(axis=0)
        elif values.shape[-1] == num_channels:
            raw_heatmap = values.sum(axis=-1)
        else:
            raw_heatmap = values.mean(axis=0)
    elif values.ndim == 2:
        raw_heatmap = values
    else:
        raise ValueError(f"Unsupported DeepSHAP output shape: {values.shape}")

    positive_heatmap = np.maximum(raw_heatmap, 0.0)
    if np.allclose(float(positive_heatmap.max()), 0.0):
        positive_heatmap = np.abs(raw_heatmap)
    return normalize_map(positive_heatmap.astype(np.float32))


class DeepSHAPExplainer:
    name = "deepshap"

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        background_data: torch.Tensor,
        *,
        check_additivity: bool = False,
    ):
        self.model = model
        self.device = device
        self.background_data = background_data.detach().to(device)
        self.check_additivity = bool(check_additivity)
        self.explanation_model = _build_explanation_model(model, device)

    def explain(
        self,
        image_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> ExplanationResult:
        try:
            import shap
        except ImportError as exc:
            raise ImportError(
                "DeepSHAP requires the `shap` package. Install it with `pip install shap`."
            ) from exc

        self.model.eval()
        self.explanation_model.eval()
        probs = predict_probabilities(self.model, image_tensor, self.device).squeeze(0).numpy()
        predicted_class = int(probs.argmax())
        if target_class is None:
            target_class = predicted_class
        confidence = float(probs[target_class])

        wrapped_model = _TargetProbabilityWrapper(self.explanation_model, target_class).to(self.device)
        wrapped_model.eval()
        explainer = shap.DeepExplainer(wrapped_model, self.background_data)
        input_batch = image_tensor.unsqueeze(0).to(self.device)
        shap_values = explainer.shap_values(
            input_batch,
            check_additivity=self.check_additivity,
        )
        heatmap = _shap_values_to_heatmap(shap_values, num_channels=int(image_tensor.shape[0]))

        return ExplanationResult(
            method=self.name,
            heatmap=heatmap,
            target_class=target_class,
            predicted_class=predicted_class,
            confidence=confidence,
            metadata={
                "background_size": int(self.background_data.shape[0]),
                "check_additivity": self.check_additivity,
                "expected_value": _expected_value_to_float(getattr(explainer, "expected_value", None)),
                "explained_output": "target_probability",
            },
        )
