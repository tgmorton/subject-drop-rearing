import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union

_LOGGERS_CREATED = set()  # avoid duplicate handlers in multiprocessing


def setup_logging(
    name: str,
    experiment: str = "global",
    log_dir: Union[str, Path] = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Initialize (or re-use) a logger with a consistent format and file location.

    Each experiment gets its own sub-folder: logs/<experiment>/
    File names: <script-name>_<YYYYMMDD_HHMMSS>.log
    """
    if name in _LOGGERS_CREATED:  # already configured – just return it
        return logging.getLogger(name)

    log_dir = Path(log_dir) / experiment
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{name.split('.')[-1]}_{timestamp}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    file_handler = logging.FileHandler(log_dir / file_name)
    stream_handler = logging.StreamHandler(sys.stdout)

    for h in (file_handler, stream_handler):
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False  # children still bubble up, avoids double prints

    _LOGGERS_CREATED.add(name)
    return logger 