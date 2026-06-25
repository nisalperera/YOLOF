import os
import sys
import logging
import inspect

cwd = os.getcwd()
if "yolof_soup" in cwd:
    cwd = cwd.split("yolof_soup")[0]
    

class EmptyHandler(logging.Handler):
    def emit(self, record):
        pass

class PrintLogger():
    def info(self, msg): print(msg)
    def debug(self, msg): print(msg)
    def warn(self, msg): print(msg)
    def error(self, msg): print(msg)
    def critical(self, msg): print(msg)

def setup_logging(level=logging.INFO, filename="logs.log", use_stdout=True):
    """
    Setup logging with file, stdout, and optional ClearML handlers.
    
    Args:
        level: Logging level (default: logging.INFO)
        filename: Log file path (default: "logs.log")
        use_stdout: Whether to log to stdout (default: True)
    Returns:
        Configured logger instance
    """
    # Get or create root logger

    caller = inspect.stack()[1]

    logger_name = os.path.basename(caller.filename).replace(".py", "")
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatters
    formatter = logging.Formatter("%(levelname)-8s | %(name)s | %(message)s")
    
    # Add stdout handler if requested
    if use_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(level)
        stdout_handler.setFormatter(formatter)
        logger.addHandler(stdout_handler)
    
    os.makedirs(os.path.dirname(os.path.join(cwd, "logs", logger_name, filename)), exist_ok=True)

    # Add file handler
    file_handler = logging.FileHandler(os.path.join(cwd, "logs", logger_name, filename))
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # If no handlers were added, add an empty handler to avoid "No handlers could be found" warnings
    if not logger.handlers:
        logger.addHandler(EmptyHandler())

    return logger