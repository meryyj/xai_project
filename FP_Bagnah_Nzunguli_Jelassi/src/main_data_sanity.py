# src/main_data_sanity.py

from src.utils.config import load_config
from src.utils.logging import setup_logging
from src.data.datasets import build_dataloaders


def main():
    logger = setup_logging(name="data_sanity", reset_handlers=True)
    cfg = load_config("configs/main_cnn_template.yaml")
    logger.info("Loaded config")

    loaders = build_dataloaders(cfg.data)
    train_loader = loaders["train"]

    # fetch one batch
    images, labels = next(iter(train_loader))
    logger.info(f"Batch images shape: {images.shape}")
    logger.info(f"Batch labels shape: {labels.shape}")
    logger.info(f"Labels (first 10): {labels[:10]}")

if __name__ == "__main__":
    main()
