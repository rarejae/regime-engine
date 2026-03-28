"""loguru configuration for the Macro Regime Engine."""

from __future__ import annotations

import os
import sys

import yaml
from loguru import logger

_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "settings.yaml")


def setup_logging(level: str | None = None) -> None:
    """
    Configure loguru with console and rotating file handlers.

    Args:
        level: Override log level string (DEBUG/INFO/WARNING/ERROR).
               If None, reads from config/settings.yaml or LOG_LEVEL env var.
    """
    with open(_SETTINGS_PATH) as f:
        settings = yaml.safe_load(f)

    log_cfg = settings.get("logging", {})
    effective_level = level or os.getenv("LOG_LEVEL", log_cfg.get("level", "INFO"))
    log_format = log_cfg.get(
        "format",
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    log_file = log_cfg.get("file", "logs/regime_engine.log")
    rotation = log_cfg.get("rotation", "10 MB")
    retention = log_cfg.get("retention", "30 days")

    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=effective_level,
        format=log_format,
        colorize=True,
    )

    # Rotating file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.add(
        log_file,
        level="DEBUG",  # always write debug to file
        format=log_format,
        rotation=rotation,
        retention=retention,
        colorize=False,
        enqueue=True,  # thread-safe
    )

    logger.debug(f"Logging configured: console={effective_level}, file=DEBUG")
