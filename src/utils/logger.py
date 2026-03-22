# -*- coding: utf-8 -*-
"""
Logging configuration for the gesture recognition system.
Provides structured logging with file and console output.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Logger instances cache
_loggers: dict = {}


def setup_logger(
    name: str = "gesture_recognition",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up and configure a logger.
    
    Args:
        name: Logger name
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Path to log file (optional)
        log_format: Format string for log messages
        date_format: Format string for timestamps
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    # Check if logger already exists
    if name in _loggers:
        return _loggers[name]
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []  # Clear existing handlers
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt=date_format)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Cache the logger
    _loggers[name] = logger
    
    return logger


def get_logger(name: str = "gesture_recognition") -> logging.Logger:
    """
    Get a logger instance. Creates one if it doesn't exist.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if name not in _loggers:
        return setup_logger(name)
    return _loggers[name]


def create_session_log_file() -> str:
    """
    Create a unique log file path for the current session.
    
    Returns:
        Path to the log file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    return str(log_dir / f"session_{timestamp}.log")


# Convenience logging functions
def log_info(message: str, logger_name: str = "gesture_recognition") -> None:
    """Log an info message."""
    get_logger(logger_name).info(message)


def log_warning(message: str, logger_name: str = "gesture_recognition") -> None:
    """Log a warning message."""
    get_logger(logger_name).warning(message)


def log_error(message: str, logger_name: str = "gesture_recognition") -> None:
    """Log an error message."""
    get_logger(logger_name).error(message)


def log_debug(message: str, logger_name: str = "gesture_recognition") -> None:
    """Log a debug message."""
    get_logger(logger_name).debug(message)


def log_exception(message: str, logger_name: str = "gesture_recognition") -> None:
    """Log an exception with traceback."""
    get_logger(logger_name).exception(message)
