"""Structured logging setup with Rich console and rotating file handlers."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_level: str = "INFO", logs_dir: Path | None = None) -> logging.Logger:
    """Configure application-wide logging.

    Args:
        log_level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        logs_dir: Directory for log files. If None, only console logging is used.

    Returns:
        The root 'market_data' logger.
    """
    logger = logging.getLogger("market_data")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplicates on re-init
    logger.handlers.clear()

    # --- Console handler ---
    console_fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_fmt)
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.addHandler(console_handler)

    # --- File handler (rotating) ---
    if logs_dir is not None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "market_data.log"

        file_fmt = logging.Formatter(
            fmt="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_fmt)
        file_handler.setLevel(logging.DEBUG)  # File always captures DEBUG+
        logger.addHandler(file_handler)

    # Prevent propagation to the root logger
    logger.propagate = False

    return logger
