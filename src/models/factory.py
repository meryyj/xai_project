from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .main_cnn import build_model as build_main_cnn
from .shallow_cnn import ShallowCNN


def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """
    Small model factory so training/evaluation code can work with multiple baselines.
    """
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})

    model_name = str(model_cfg.get("name", "main_cnn")).lower()
    in_channels = int(dataset_cfg.get("channels", model_cfg.get("in_channels", 1)))
    num_classes = int(model_cfg.get("num_classes", 7))

    if model_name in {"shallow", "shallow_cnn", "baseline_cnn"}:
        hidden_dim = int(model_cfg.get("hidden_dim", 128))
        dropout = float(model_cfg.get("dropout_head", model_cfg.get("dropout", 0.5)))
        return ShallowCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    return build_main_cnn(cfg)
