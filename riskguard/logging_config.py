"""Logging configuration for RiskGuard."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "riskguard", level: int = logging.INFO) -> logging.Logger:
    """Return a configured module logger without duplicating handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
