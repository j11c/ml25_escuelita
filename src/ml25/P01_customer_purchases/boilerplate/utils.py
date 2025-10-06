import logging
import os
from datetime import datetime
from pathlib import Path


CURRENT_FILE = Path(__file__).resolve()
LOGS_DIR = CURRENT_FILE.parent / "logs"


def setup_logger(name: str, log_dir: str = "./logs") -> logging.Logger:
    """
    Setup a logger that writes to console and file.
    """
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    log_file = LOGS_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers (avoid duplicates)
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
