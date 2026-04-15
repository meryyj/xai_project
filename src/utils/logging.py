# src/utils/logging.py

from __future__ import annotations

import logging
from pathlib import Path


def _resolve_log_file(
    log_file: str | Path,
    name: str | None,
) -> tuple[Path, str]:
    """
    Resolve a log destination while staying backward-compatible with older calls.

    Supported patterns:
    - setup_logging("experiments/logs/train.log")
    - setup_logging("sanity") -> experiments/logs/sanity.log
    - setup_logging(name="data_sanity") -> experiments/logs/data_sanity.log
    """
    if name is not None:
        return Path("experiments/logs") / f"{name}.log", name

    path = Path(log_file)
    if path.parent == Path(".") and path.suffix == "":
        return Path("experiments/logs") / f"{path.name}.log", path.name

    logger_name = path.stem or "root"
    return path, logger_name


def setup_logging(
    log_file: str | Path = "experiments/logs/train.log",
    *,
    name: str | None = None,
    reset_handlers: bool = False,
) -> logging.Logger:
    """
    Initialize Python logging with a single file handler + console.

    Returns the logger associated with the resolved log name so callers can use:
        logger = setup_logging(name="sanity", reset_handlers=True)
    """
    log_path, logger_name = _resolve_log_file(log_file, name)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=reset_handlers,
    )

    root_logger = logging.getLogger()
    root_logger.info(f"Logging initialized. Writing to: {log_path}")
    return logging.getLogger(logger_name)
