import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Union, Dict, Optional

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


def setup_multi_logging(
    experiment: str = "global",
    log_dir: Union[str, Path] = "logs",
    level: int = logging.INFO,
) -> Dict[str, logging.Logger]:
    """
    Set up multiple loggers for different types of output.
    
    Returns a dictionary with loggers for:
    - 'main': General output and info messages
    - 'errors': Error and warning messages
    - 'ablation': Detailed ablation reports and debug info
    - 'progress': Progress updates and status messages
    """
    log_dir = Path(log_dir) / experiment
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    loggers = {}
    
    # Main logger (general output)
    main_logger = logging.getLogger(f"{experiment}.main")
    main_logger.setLevel(level)
    main_logger.propagate = False
    
    main_file_handler = logging.FileHandler(log_dir / f"main_{timestamp}.log")
    main_stream_handler = logging.StreamHandler(sys.stdout)
    
    for h in (main_file_handler, main_stream_handler):
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        main_logger.addHandler(h)
    
    loggers['main'] = main_logger
    
    # Error logger (errors and warnings only)
    error_logger = logging.getLogger(f"{experiment}.errors")
    error_logger.setLevel(logging.WARNING)
    error_logger.propagate = False
    
    error_file_handler = logging.FileHandler(log_dir / f"errors_{timestamp}.log")
    error_stream_handler = logging.StreamHandler(sys.stderr)
    
    for h in (error_file_handler, error_stream_handler):
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        error_logger.addHandler(h)
    
    loggers['errors'] = error_logger
    
    # Ablation logger (detailed ablation reports)
    ablation_logger = logging.getLogger(f"{experiment}.ablation")
    ablation_logger.setLevel(logging.DEBUG)
    ablation_logger.propagate = False
    
    ablation_file_handler = logging.FileHandler(log_dir / f"ablation_{timestamp}.log")
    
    for h in (ablation_file_handler,):
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        ablation_logger.addHandler(h)
    
    loggers['ablation'] = ablation_logger
    
    # Progress logger (progress updates)
    progress_logger = logging.getLogger(f"{experiment}.progress")
    progress_logger.setLevel(logging.INFO)
    progress_logger.propagate = False
    
    progress_file_handler = logging.FileHandler(log_dir / f"progress_{timestamp}.log")
    progress_stream_handler = logging.StreamHandler(sys.stdout)
    
    for h in (progress_file_handler, progress_stream_handler):
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        progress_logger.addHandler(h)
    
    loggers['progress'] = progress_logger
    
    return loggers 