from __future__ import annotations

import logging
from pathlib import Path


LOGGER_NAME = "srm_steganalysis"


def configure_logging(log_file: str | Path, verbose: bool = False) -> logging.Logger:
    """Configure timestamped console and persistent file logging for one CLI run."""
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info("Log file: %s", path.resolve())
    return logger


def elapsed_text(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.3f} s"
    minutes, remainder = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)} min {remainder:.1f} s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)} h {int(minutes)} min {remainder:.1f} s"
