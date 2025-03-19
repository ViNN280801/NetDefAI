# /model_training/__init__.py

"""
Common modules for model training including the universal trainer and device management.
"""

from .device_manager import (
    ComputeDevice,
    DeviceManager,
    get_device_manager,
)
from .universal_trainer import (
    TextDataset,
    TextClassifierANN,
    UniversalTrainer,
    menu,
)

__all__ = [
    "ComputeDevice",
    "DeviceManager",
    "get_device_manager",
    "TextDataset",
    "TextClassifierANN",
    "UniversalTrainer",
    "menu",
]
