# File name: logger.py
# Description: Logger for the Ad Heatmap API
#              This module provides a logger for the Ad Heatmap API.
#              It includes methods for logging messages at different levels.
#              It also provides a singleton pattern for logging configuration

import os
import sys
import logging
import inspect
from colorlog import ColoredFormatter

SUPPORTED_LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Logger:
    _loggers = {}  # Dictionary with loggers by module name

    def _configure_logger(self, log_level, module_name, log_file=None):
        """Creates and configures a logger for a specific module."""
        if module_name in self._loggers:
            return self._loggers[module_name]  # Already configured

        # Check logging level
        if log_level not in SUPPORTED_LOG_LEVELS:
            print(
                f"⚠️ WARNING: Invalid LOG_LEVEL '{log_level}', falling back to 'INFO'.",
                file=sys.stderr,
            )
            log_level = "INFO"

        numeric_level = getattr(logging, log_level)

        # Configure log format with colorlog for console output
        console_formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s | %(levelname)s - %(name)s : %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "white",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )

        # Plain formatter for file output
        file_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s - %(name)s : %(message)s",
            datefmt=None,
            style="%",
        )

        # Create new logger
        logger = logging.getLogger(module_name)
        logger.setLevel(numeric_level)
        logger.propagate = False  # Don't propagate to parent loggers

        # Remove existing handlers if any
        if logger.handlers:
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

        # Add console handler with color formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)

        # Add file handler if log_file is provided
        if log_file:
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)

        print(
            f"ℹ️ INFO: Logger initialized for '{module_name}' with level: '{log_level}'",
            file=sys.stdout,
        )
        if log_file:
            print(f"ℹ️ INFO: Logging to file: '{log_file}'", file=sys.stdout)

        self._loggers[module_name] = logger
        return logger

    def _get_caller_module(self):
        """Determines the name of the module that called the logger."""
        stack = inspect.stack()

        # Getting module name from stack trace
        # 2 because:
        #   0 is this function,
        #   1 is the function that called this one,
        #   2 is the function that called the one that called this one
        #   ... and so on
        module = inspect.getmodule(stack[2][0])
        return module.__name__ if module else "unknown"

    def __init__(self, log_level, caller_module=None, log_file=None, level=None):
        """
        Initialize a logger.

        Args:
            log_level (str): The log level as a string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            caller_module (str, optional): The module name to use for the logger.
                If None, it will be determined automatically.
            log_file (str, optional): Path to a log file. If provided,
                logs will be written to this file.
            level (int, optional): Log level as a logging constant.
                This is an alternative to log_level and takes precedence if provided.
        """
        # Support both string log_level and numeric level
        if level is not None:
            log_level = logging.getLevelName(level)

        if not caller_module:
            caller_module = self._get_caller_module()

        self.logger = self._configure_logger(log_level, caller_module, log_file)

    def debug(self, message, *args, **kwargs):
        self.logger.debug(message, *args, **kwargs)

    def info(self, message, *args, **kwargs):
        self.logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        self.logger.warning(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        self.logger.error(message, *args, **kwargs, exc_info=True)

    def critical(self, message, *args, **kwargs):
        self.logger.critical(message, *args, **kwargs, exc_info=True)
