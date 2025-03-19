#!/usr/bin/env python3
"""
Helper module for Path Traversal attack dataset generation.

This module provides functions to generate datasets for Path Traversal attack detection.
"""

import os
import sys
import random
import pandas as pd
import numpy as np
from typing import List
from src.logger import dataset_generator_logger as logger

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Sample file paths for normal requests
normal_PATHS = [
    "/home/user/documents/report.pdf",
    "/var/www/html/index.html",
    "/etc/nginx/sites-available/default",
    "/usr/share/doc/python/index.html",
    "/home/user/images/profile.jpg",
    "/var/log/apache2/access.log",
    "/opt/application/config.json",
    "/home/user/downloads/file.zip",
    "/mnt/data/backup.tar.gz",
    "/home/user/projects/website/css/style.css",
]

# Sample file extensions
FILE_EXTENSIONS = [
    ".txt",
    ".html",
    ".php",
    ".jsp",
    ".asp",
    ".json",
    ".xml",
    ".pdf",
    ".doc",
    ".jpg",
    ".png",
    ".zip",
    ".tar.gz",
    ".log",
    ".csv",
]

# Sample directories
DIRECTORIES = [
    "/home/user",
    "/var/www",
    "/etc",
    "/usr/share",
    "/opt",
    "/var/log",
    "/mnt/data",
    "/home/admin",
    "/srv",
    "/tmp",
]


def generate_normal_path() -> str:
    """
    Generate a normal file path.

    Returns:
        A string representing a normal file path
    """
    # Either use a predefined normal path or generate a new one
    if random.random() < 0.7:
        return random.choice(normal_PATHS)

    # Generate a random path
    directory = random.choice(DIRECTORIES)

    # Add subdirectories (0-3 levels)
    for _ in range(random.randint(0, 3)):
        directory = os.path.join(directory, f"folder{random.randint(1, 100)}")

    # Add filename
    filename = f"file{random.randint(1, 1000)}"
    extension = random.choice(FILE_EXTENSIONS)

    # Return normalized path with forward slashes
    return os.path.join(directory, filename + extension).replace("\\", "/")


def generate_malicious_path(attack_patterns: List[str]) -> str:
    """
    Generate a malicious file path using Path Traversal attack patterns.

    Args:
        attack_patterns: List of Path Traversal attack patterns

    Returns:
        A string representing a malicious file path
    """
    # Choose a random attack pattern
    attack_pattern = random.choice(attack_patterns)

    # There are several ways to construct a path traversal
    attack_type = random.randint(1, 4)

    if attack_type == 1:
        # Direct use of attack pattern
        return attack_pattern
    elif attack_type == 2:
        # Combine attack pattern with a normal-looking path
        return f"{attack_pattern}/file.txt"
    elif attack_type == 3:
        # Use attack pattern in the middle of a path
        return f"/var/www/{attack_pattern}/file.html"
    else:
        # Use attack pattern with URL encoding
        # Simple encoding example - in reality would use urllib.parse
        encoded_pattern = attack_pattern.replace(".", "%2E").replace("/", "%2F")
        return f"/home/user/{encoded_pattern}/config.json"


def generate_mixed_dataset(
    attack_patterns: List[str],
    num_normal: int = 2500,
    num_malicious: int = 2500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a mixed dataset of normal and malicious file paths.

    Args:
        attack_patterns: List of Path Traversal attack patterns
        num_normal: Number of normal samples to generate
        num_malicious: Number of malicious samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing the mixed dataset
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Generate normal paths
    normal_paths = []
    for _ in range(num_normal):
        path = generate_normal_path()
        normal_paths.append({"path": path, "label": 0})

    # Generate malicious paths
    malicious_paths = []
    for _ in range(num_malicious):
        path = generate_malicious_path(attack_patterns)
        malicious_paths.append({"path": path, "label": 1})

    # Combine and shuffle the dataset
    dataset = pd.DataFrame(normal_paths + malicious_paths)
    dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Generated Path Traversal dataset with {num_normal} normal and {num_malicious} malicious samples"
    )

    return dataset
