"""
utils/logger.py
───────────────
Centralised logging with coloured console output and optional file sink.
"""

import logging
import os
import sys
from datetime import datetime


_LOG_COLOURS = {
    "DEBUG":    "\033[36m",   # cyan
    "INFO":     "\033[32m",   # green
    "WARNING":  "\033[33m",   # yellow
    "ERROR":    "\033[31m",   # red
    "CRITICAL": "\033[1;31m", # bold red
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """Adds ANSI colour codes to log-level names."""

    def format(self, record: logging.LogRecord) -> str:
        colour = _LOG_COLOURS.get(record.levelname, "")
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for *name*."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already set up

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    # ── Console handler ───────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        _ColourFormatter("%(asctime)s │ %(levelname)s │ %(name)s │ %(message)s",
                         datefmt="%H:%M:%S")
    )
    logger.addHandler(console)

    # ── File handler (optional) ───────────────────────────
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"orchestrator_{datetime.now():%Y%m%d}.log"),
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s")
    )
    logger.addHandler(file_handler)

    return logger
