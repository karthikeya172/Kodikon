"""
Centralized logging utilities for the Kodikon system.

Provides consistent logging configuration, formatters, and handlers
across all modules for better observability and debugging.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


# Color codes for console output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'


class ColoredFormatter(logging.Formatter):
    """
    Formatter that adds colors to log levels for better readability.
    """

    COLORS = {
        'DEBUG': Colors.CYAN,
        'INFO': Colors.GREEN,
        'WARNING': Colors.YELLOW,
        'ERROR': Colors.RED,
        'CRITICAL': Colors.RED + Colors.BOLD,
    }

    def format(self, record):
        """Format log record with colors."""
        if record.levelname in self.COLORS:
            level_color = self.COLORS[record.levelname]
            record.levelname = f"{level_color}{record.levelname}{Colors.RESET}"

        return super().format(record)


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    enable_console: bool = True,
    enable_file: bool = True,
    colored_output: bool = True,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup and configure a logger with console and file handlers.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files. If None, uses default './logs'
        enable_console: Enable console output
        enable_file: Enable file output
        colored_output: Use colored console output
        max_bytes: Maximum size for log file before rotation (default 10MB)
        backup_count: Number of backup log files to keep

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__, level=logging.DEBUG)
        >>> logger.info("Application started")
    """
    logger = logging.getLogger(name)

    # Only configure if not already configured
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Create formatters
    detailed_format = (
        '%(asctime)s - %(name)s - %(levelname)s - '
        '[%(filename)s:%(lineno)d] - %(message)s'
    )
    simple_format = '%(levelname)s - %(message)s'

    detailed_formatter = logging.Formatter(detailed_format)
    console_formatter = (
        ColoredFormatter(simple_format) if colored_output
        else logging.Formatter(simple_format)
    )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if enable_file:
        log_dir = Path(log_dir or 'logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{name.replace('.', '_')}.log"

        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with the given name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def configure_root_logger(
    level: int = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Configure the root logger for the entire application.

    Args:
        level: Logging level
        log_dir: Directory for log files

    Returns:
        Root logger

    Example:
        >>> setup_logger('kodikon', level=logging.DEBUG, log_dir='./logs')
    """
    return setup_logger(
        'kodikon',
        level=level,
        log_dir=log_dir,
        colored_output=True
    )


class LogContext:
    """
    Context manager for temporary logging level changes.

    Example:
        >>> logger = setup_logger(__name__)
        >>> with LogContext(logger, logging.DEBUG):
        ...     logger.debug("This debug message will be logged")
    """

    def __init__(self, logger: logging.Logger, level: int):
        """
        Initialize log context.

        Args:
            logger: Logger instance
            level: Temporary logging level
        """
        self.logger = logger
        self.original_level = logger.level
        self.new_level = level

    def __enter__(self):
        """Enter context."""
        self.logger.setLevel(self.new_level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        self.logger.setLevel(self.original_level)


def log_exception(logger: logging.Logger, exception: Exception) -> None:
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance
        exception: Exception to log

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, e)
    """
    logger.exception(f"Exception occurred: {exception}")


def create_performance_logger(
    name: str = "performance",
    log_dir: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """
    Create a specialized logger for performance metrics.

    Args:
        name: Logger name
        log_dir: Directory for log files

    Returns:
        Performance logger instance

    Example:
        >>> perf_logger = create_performance_logger()
        >>> perf_logger.info("Operation completed in 1.23 seconds")
    """
    return setup_logger(
        f"kodikon.{name}",
        level=logging.INFO,
        log_dir=log_dir,
        colored_output=False
    )


def log_operation(
    logger: logging.Logger,
    operation_name: str,
    result: bool,
    message: str = "",
    details: Optional[dict] = None
) -> None:
    """
    Log the result of an operation.

    Args:
        logger: Logger instance
        operation_name: Name of operation
        result: Whether operation succeeded
        message: Optional message
        details: Optional details dictionary

    Example:
        >>> log_operation(
        ...     logger,
        ...     "video_processing",
        ...     result=True,
        ...     message="Processed 100 frames",
        ...     details={"frames": 100, "time": 5.2}
        ... )
    """
    status = "SUCCESS" if result else "FAILED"
    log_func = logger.info if result else logger.error

    msg = f"[{operation_name}] {status}"
    if message:
        msg += f": {message}"
    if details:
        msg += f" | {details}"

    log_func(msg)


def get_log_statistics(log_file: Union[str, Path]) -> dict:
    """
    Get statistics from a log file.

    Args:
        log_file: Path to log file

    Returns:
        Dictionary with log statistics

    Example:
        >>> stats = get_log_statistics('logs/kodikon.log')
        >>> print(f"Errors: {stats['ERROR']}, Warnings: {stats['WARNING']}")
    """
    log_file = Path(log_file)

    if not log_file.exists():
        return {}

    stats = {
        'DEBUG': 0,
        'INFO': 0,
        'WARNING': 0,
        'ERROR': 0,
        'CRITICAL': 0,
        'TOTAL': 0
    }

    try:
        with open(log_file, 'r') as f:
            for line in f:
                stats['TOTAL'] += 1
                for level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                    if f' - {level} - ' in line:
                        stats[level] += 1
                        break
    except Exception as e:
        logging.warning(f"Failed to read log statistics: {e}")

    return stats


# Convenience function
def log_info(message: str, logger_name: str = "kodikon") -> None:
    """
    Quick logging helper.

    Args:
        message: Message to log
        logger_name: Logger name
    """
    logging.getLogger(logger_name).info(message)


def log_error(message: str, logger_name: str = "kodikon") -> None:
    """
    Quick error logging helper.

    Args:
        message: Message to log
        logger_name: Logger name
    """
    logging.getLogger(logger_name).error(message)


def log_warning(message: str, logger_name: str = "kodikon") -> None:
    """
    Quick warning logging helper.

    Args:
        message: Message to log
        logger_name: Logger name
    """
    logging.getLogger(logger_name).warning(message)


def log_debug(message: str, logger_name: str = "kodikon") -> None:
    """
    Quick debug logging helper.

    Args:
        message: Message to log
        logger_name: Logger name
    """
    logging.getLogger(logger_name).debug(message)
