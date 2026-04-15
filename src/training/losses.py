# src/training/losses.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


LossName = Literal["cross_entropy", "weighted_cross_entropy", "focal"]


@dataclass
class LossConfig:
    name: LossName = "cross_entropy"
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0  # scaling factor, not class-balancing alpha


class LabelSmoothedCrossEntropy(nn.Module):
    """
    Cross-entropy with optional label smoothing.
    """

    def __init__(
        self,
        num_classes: int,
        smoothing: float = 0.0,
        weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        log_probs = F.log_softmax(logits, dim=-1)
        with torch.no_grad():
            # one-hot with smoothing
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        if self.weight is not None:
            # expand weights to batch shape
            w = self.weight.unsqueeze(0)  # (1, C)
            loss = (-true_dist * log_probs * w).sum(dim=1)
        else:
            loss = (-true_dist * log_probs).sum(dim=1)
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Standard focal loss for multi-class classification.

    gamma: focusing parameter
    alpha: global scaling factor (not per-class alpha here)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 1.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
        num_classes: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert reduction in ("none", "mean", "sum")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        if self.label_smoothing > 0.0 and self.num_classes is not None:
            with torch.no_grad():
                smooth = self.label_smoothing
                true_dist = torch.zeros_like(log_probs)
                true_dist.fill_(smooth / (self.num_classes - 1))
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smooth)
        else:
            with torch.no_grad():
                true_dist = torch.zeros_like(log_probs)
                true_dist.scatter_(1, targets.unsqueeze(1), 1.0)

        pt = (probs * true_dist).sum(dim=1)  # prob of the true class
        log_pt = (log_probs * true_dist).sum(dim=1)

        focal_term = (1.0 - pt) ** self.gamma
        loss = -self.alpha * focal_term * log_pt

        if self.weight is not None:
            # per-class weights
            w = (self.weight[targets]).to(loss.device)
            loss = loss * w

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(class_counts: Sequence[int]) -> torch.Tensor:
    """
    Compute simple inverse-frequency class weights.
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    total = counts.sum()
    num_classes = len(class_counts)
    # w_i = N / (K * n_i)
    weights = total / (num_classes * counts.clamp(min=1.0))
    return weights


def build_loss(
    num_classes: int,
    loss_cfg: Optional[dict] = None,
    class_counts: Optional[Sequence[int]] = None,
) -> nn.Module:
    """
    Build loss function from config + optional class counts.

    loss_cfg can contain:
      - name: "cross_entropy", "weighted_cross_entropy", or "focal"
      - label_smoothing: float
      - focal_gamma: float
      - focal_alpha: float
    """

    if loss_cfg is None:
        loss_cfg = {}

    name: LossName = loss_cfg.get("name", "cross_entropy")
    smoothing: float = loss_cfg.get("label_smoothing", 0.0)
    focal_gamma: float = loss_cfg.get("focal_gamma", 2.0)
    focal_alpha: float = loss_cfg.get("focal_alpha", 1.0)

    weight = None
    if class_counts is not None:
        weight = compute_class_weights(class_counts)

    if name == "cross_entropy":
        if smoothing > 0.0:
            return LabelSmoothedCrossEntropy(
                num_classes=num_classes, smoothing=smoothing, weight=weight
            )
        else:
            return nn.CrossEntropyLoss(weight=weight)

    if name == "weighted_cross_entropy":
        # same as CE but forcing weights if available
        return nn.CrossEntropyLoss(weight=weight)

    if name == "focal":
        return FocalLoss(
            gamma=focal_gamma,
            alpha=focal_alpha,
            weight=weight,
            reduction="mean",
            label_smoothing=smoothing,
            num_classes=num_classes,
        )

    # fallback
    return nn.CrossEntropyLoss(weight=weight)
