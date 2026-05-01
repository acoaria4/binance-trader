"""
utils/logger.py
Shared logging setup using Python's logging + Rich for coloured console output.
"""
import logging
import os
from rich.logging import RichHandler
from config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger   # already configured

    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    logger.setLevel(level)

    # Console handler (Rich)
    console = RichHandler(rich_tracebacks=True, markup=True)
    console.setLevel(level)
    logger.addHandler(console)

    # File handler
    os.makedirs(os.path.dirname(settings.LOG_FILE), exist_ok=True)
    file_h = logging.FileHandler(settings.LOG_FILE)
    file_h.setLevel(level)
    file_h.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))
    logger.addHandler(file_h)

    return logger
