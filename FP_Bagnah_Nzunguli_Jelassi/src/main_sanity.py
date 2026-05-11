# src/main_sanity.py

from __future__ import annotations

import torch

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.datasets import build_dataloaders
from src.models.factory import build_model_from_config
from src.utils.logging import setup_logging


def main():
    logger = setup_logging(name="sanity", reset_handlers=True)

    cfg = load_config("configs/main_cnn_v1.yaml")
    logger.info("Loaded config")

    seed = cfg.get("seed", 42)
    set_seed(seed)

    loaders = build_dataloaders(cfg.data)
    train_loader = loaders["train"]

    # Build model from config
    model = build_model_from_config(cfg.data)
    logger.info("Model:\n%s", model)

    # Grab one batch
    images, labels = next(iter(train_loader))
    logger.info("Batch images shape: %s", tuple(images.shape))
    logger.info("Batch labels shape: %s", tuple(labels.shape))

    # Forward pass
    with torch.no_grad():
        logits = model(images)
    logger.info("Logits shape: %s", tuple(logits.shape))


if __name__ == "__main__":
    main()
