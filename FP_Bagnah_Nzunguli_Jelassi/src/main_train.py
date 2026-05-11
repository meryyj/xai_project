# src/main_train.py

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.utils.seed import set_seed
from src.training.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train main CNN on RAF-DB")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/main_cnn_template.yaml",
        help="Path to YAML config file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Ensure logs/checkpoints dirs exist
    Path("experiments/logs").mkdir(parents=True, exist_ok=True)
    Path("experiments/checkpoints").mkdir(parents=True, exist_ok=True)

    setup_logging("experiments/logs/train.log")
    logger = logging.getLogger("train")

    cfg = load_config(args.config)
    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Config content:\n{cfg}")

    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    logger.info(f"Seed set to {seed}")

    train_model(cfg)


if __name__ == "__main__":
    main()
