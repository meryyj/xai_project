# src/training/schedulers.py

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    StepLR,
)


def build_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Optional[Dict[str, Any]],
    num_epochs: int,
    steps_per_epoch: int,
):
    """
    Build LR scheduler from config.

    Supported:
      - name: "cosine_annealing"
          uses T_max, eta_min
      - name: "reduce_on_plateau"
          uses factor, patience, min_lr
      - name: "one_cycle"
          uses max_lr
      - name: "step_lr"
          uses step_size, gamma
      - name: "none" or missing -> returns None
    """
    if scheduler_cfg is None:
        return None

    name = scheduler_cfg.get("name", "none").lower()

    if name == "none":
        return None

    if name == "cosine_annealing":
        t_max = scheduler_cfg.get("T_max", num_epochs)
        eta_min = scheduler_cfg.get("eta_min", 1e-5)
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    if name == "reduce_on_plateau":
        factor = scheduler_cfg.get("factor", 0.1)
        patience = scheduler_cfg.get("patience", 10)
        min_lr = scheduler_cfg.get("min_lr", 1e-6)
        return ReduceLROnPlateau(
            optimizer,
            mode="max",  # assume we track val_macro_f1
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    if name == "one_cycle":
        max_lr = scheduler_cfg.get("max_lr", scheduler_cfg.get("lr", 1e-3))
        total_steps = num_epochs * steps_per_epoch
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
        )

    if name == "step_lr":
        step_size = scheduler_cfg.get("step_size", 30)
        gamma = scheduler_cfg.get("gamma", 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Unknown name -> no scheduler
    return None
