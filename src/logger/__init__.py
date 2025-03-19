# /src/logger/__init__.py

"""
Logger module for the attack detection system.

This module provides loggers for different components of the system.
"""

from .logger import Logger, SUPPORTED_LOG_LEVELS
from .logger_settings import (
    get_dataset_logger,
    get_model_logger,
    path_traversal_dataset_logger,
    sql_injections_dataset_logger,
    xss_dataset_logger,
    dos_dataset_logger,
    path_traversal_model_logger,
    sql_injections_model_logger,
    xss_model_logger,
    dos_model_logger,
    universal_trainer_logger,
    dataset_generator_logger,
    device_manager_logger,
)

__all__ = [
    "Logger",
    "SUPPORTED_LOG_LEVELS",
    "get_dataset_logger",
    "get_model_logger",
    "path_traversal_dataset_logger",
    "sql_injections_dataset_logger",
    "xss_dataset_logger",
    "dos_dataset_logger",
    "path_traversal_model_logger",
    "sql_injections_model_logger",
    "xss_model_logger",
    "dos_model_logger",
    "universal_trainer_logger",
    "dataset_generator_logger",
    "device_manager_logger",
]
