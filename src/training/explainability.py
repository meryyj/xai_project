from __future__ import annotations

from typing import Any, Dict

import torch


def explainability_enabled(cfg: Dict[str, Any] | None) -> bool:
    return bool((cfg or {}).get("enabled", False))


def build_face_prior_mask(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    center_x: float = 0.5,
    center_y: float = 0.52,
    radius_x: float = 0.34,
    radius_y: float = 0.42,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(0.0, 1.0, steps=height, device=device, dtype=dtype),
        torch.linspace(0.0, 1.0, steps=width, device=device, dtype=dtype),
        indexing="ij",
    )
    ellipse = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2) <= 1.0
    return ellipse.to(dtype=dtype).unsqueeze(0).unsqueeze(0)


def compute_explainability_loss(
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    labels: torch.Tensor,
    explainability_cfg: Dict[str, Any],
) -> torch.Tensor:
    """
    Differentiable explanation-guided regularization using input-gradient saliency.

    The loss penalizes saliency mass outside a central face prior, which is a practical
    approximation for aligned FER datasets such as RAF-DB.
    """
    target_scores = outputs.gather(1, labels.view(-1, 1)).sum()
    gradients = torch.autograd.grad(
        target_scores,
        inputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    saliency = gradients.abs().mean(dim=1, keepdim=True)
    saliency = saliency / saliency.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)

    prior = build_face_prior_mask(
        inputs.shape[2],
        inputs.shape[3],
        device=inputs.device,
        dtype=inputs.dtype,
        center_x=float(explainability_cfg.get("center_x", 0.5)),
        center_y=float(explainability_cfg.get("center_y", 0.52)),
        radius_x=float(explainability_cfg.get("radius_x", 0.34)),
        radius_y=float(explainability_cfg.get("radius_y", 0.42)),
    )
    prior = prior.expand(inputs.shape[0], 1, -1, -1)

    inside_mass = (saliency * prior).sum(dim=(1, 2, 3))
    loss = 1.0 - inside_mass
    return loss.mean()
