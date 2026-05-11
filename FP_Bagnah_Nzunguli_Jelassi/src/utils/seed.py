# src/utils/seed.py

import os
import random
import logging
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # PyTorch not installed yet
    torch = None


def set_seed(seed: int = 42, deterministic_cudnn: bool = True) -> None:
    """
    Set random seeds for Python, NumPy and (optionally) PyTorch to ensure
    reproducible experiments.

    Parameters
    ----------
    seed : int
        Seed value to use across libraries.
    deterministic_cudnn : bool
        If True and PyTorch is available, makes cuDNN deterministic
        (slower but more reproducible).
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Setting global seed to {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("PyTorch cuDNN set to deterministic mode")
        else:
            logger.info("PyTorch cuDNN left in non-deterministic mode")
    else:
        logger.info("PyTorch not installed, skipping PyTorch seeding.")
