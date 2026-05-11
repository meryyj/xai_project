# src/utils/config.py

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    """
    Lightweight wrapper around a nested dictionary loaded from a YAML config file.

    You can access the underlying data via `cfg.data` or treat this as a simple
    container that travels through your training/eval code.
    """

    path: Path
    data: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Shorthand for cfg.data.get(key).
        """
        return self.data.get(key, default)

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __contains__(self, item: str) -> bool:
        return item in self.data

    def pretty(self) -> str:
        """
        Returns a YAML-formatted string of the config for logging/printing.
        """
        return yaml.safe_dump(self.data, sort_keys=False, default_flow_style=False)


def load_config(config_path: str | Path) -> Config:
    """
    Load a YAML configuration file from disk into a Config object.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML config file.

    Returns
    -------
    Config
        Config object wrapping the loaded data.
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return Config(path=config_path, data=data)


def parse_args_with_config(default_config: str = "configs/main_cnn_template.yaml") -> Config:
    """
    Convenience helper: parse a `--config` argument from CLI and load it.

    Usage:
        python -m src.training.train --config configs/main_cnn_template.yaml

    If no --config is provided, uses `default_config`.
    """
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition Training")
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to YAML configuration file.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    return cfg
