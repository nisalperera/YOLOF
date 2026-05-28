"""
Global logger module for YOLOF soup project.

This module provides a singleton global logger that can be imported and used
from any module in the project. It ensures consistent logging across all modules
with a single configuration.

Usage:
    from yolof_soup.utils.global_logger import get_logger
    
    logger = get_logger()
    logger.info("This is an info message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Optional: Configure logging at application startup
    from yolof_soup.utils.global_logger import configure_logger
    configure_logger(level=logging.DEBUG, add_file_handler=True)
"""

import os
import sys
import logging
import inspect

from pathlib import Path
from typing import Optional

# # Global logger instance
# _global_logger: Optional[logging.Logger] = None
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_logger(
    level: int = logging.INFO,
    add_file_handler: bool = False,
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Configure the global logger with specified settings.
    
    Args:
        name: Logger name (default: "YOLOF")
        level: Logging level (default: logging.INFO)
        add_file_handler: Whether to add a file handler (default: False)
        log_file: Log file name (default: "{name}.log")
        log_dir: Directory for log files (default: "logs")
    
    Returns:
        Configured logger instance
    """
    # global _global_logger
    
    caller = inspect.stack()[1]
    log_file = os.path.basename(caller.filename).replace(".py", ".log")

    # Create logger
    logger = logging.getLogger(log_file.split(".")[0])
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(processName)-14s (PID: %(process)d) | %(funcName)-35s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Add stdout handler
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    
    # Add file handler if requested
    if add_file_handler:
        if log_file is None:
            log_file = os.path.join(caller.filename.split("/")[-2], log_file)
        
        log_path = Path(PROJECT_ROOT.joinpath(log_dir)).joinpath(log_file.split(".")[0]) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# def reset_logger() -> None:
#     """Reset the global logger instance (for testing purposes)."""
#     global _global_logger
#     _global_logger = None
