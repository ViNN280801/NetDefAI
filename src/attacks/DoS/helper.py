#!/usr/bin/env python3
"""
Helper module for DoS attack dataset generation.

This module provides functions to generate datasets for DoS attack detection.
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

# Sample HTTP methods for DoS attacks
HTTP_METHODS = ["GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH"]

# Sample endpoints for normal requests
normal_ENDPOINTS = [
    "/api/users",
    "/api/products",
    "/api/orders",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/payments",
    "/api/settings",
    "/api/notifications",
    "/api/search",
    "/api/dashboard",
]

# Sample parameters for normal requests
normal_PARAMS = [
    "id",
    "name",
    "email",
    "page",
    "limit",
    "sort",
    "filter",
    "category",
    "price",
    "date",
    "status",
    "query",
    "user_id",
]


def generate_normal_request() -> str:
    """
    Generate a normal HTTP request.

    Returns:
        A string representing a normal HTTP request
    """
    method = random.choice(HTTP_METHODS)
    endpoint = random.choice(normal_ENDPOINTS)

    # Add query parameters for GET requests
    if method == "GET":
        num_params = random.randint(0, 3)
        if num_params > 0:
            params = []
            for _ in range(num_params):
                param_name = random.choice(normal_PARAMS)
                param_value = f"value{random.randint(1, 100)}"
                params.append(f"{param_name}={param_value}")
            endpoint = f"{endpoint}?{'&'.join(params)}"

    # Generate request with appropriate headers
    request = f"{method} {endpoint} HTTP/1.1\r\n"
    request += "Host: example.com\r\n"

    # Add some common headers
    if random.random() < 0.8:
        request += "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/96.0.4664.110\r\n"

    if random.random() < 0.5:
        request += "Accept: application/json\r\n"

    if method in ["POST", "PUT", "PATCH"]:
        content_type = (
            "application/json"
            if random.random() < 0.7
            else "application/x-www-form-urlencoded"
        )
        request += f"Content-Type: {content_type}\r\n"

        # Add a simple body for POST/PUT/PATCH requests
        body = ""
        if content_type == "application/json":
            num_fields = random.randint(1, 4)
            body_obj = {}
            for _ in range(num_fields):
                field = random.choice(normal_PARAMS)
                value = f"value{random.randint(1, 100)}"
                body_obj[field] = value

            # Convert to JSON-like string (simplified)
            body = "{"
            for i, (k, v) in enumerate(body_obj.items()):
                if i > 0:
                    body += ", "
                body += f'"{k}": "{v}"'
            body += "}"
        else:
            num_fields = random.randint(1, 4)
            body_parts = []
            for _ in range(num_fields):
                field = random.choice(normal_PARAMS)
                value = f"value{random.randint(1, 100)}"
                body_parts.append(f"{field}={value}")
            body = "&".join(body_parts)

        request += f"Content-Length: {len(body)}\r\n"
        request += f"\r\n{body}"
    else:
        request += "\r\n"

    return request


def generate_malicious_request(attack_patterns: List[str]) -> str:
    """
    Generate a malicious HTTP request using DoS attack patterns.

    Args:
        attack_patterns: List of DoS attack patterns

    Returns:
        A string representing a malicious HTTP request
    """
    method = random.choice(HTTP_METHODS)
    endpoint = random.choice(normal_ENDPOINTS)

    # Choose a random attack pattern
    attack_pattern = random.choice(attack_patterns)

    # There are several ways to construct a DoS attack
    attack_type = random.randint(1, 4)

    if attack_type == 1:
        # Malicious endpoint
        endpoint = f"/{attack_pattern}"
    elif attack_type == 2:
        # Malicious query parameter
        endpoint = f"{endpoint}?param={attack_pattern}"
    elif attack_type == 3:
        # Malicious header
        header_name = "X-Custom-Header"
        header_value = attack_pattern
    else:
        # Malicious body for POST/PUT
        method = random.choice(["POST", "PUT", "PATCH"])

    # Generate request with appropriate headers
    request = f"{method} {endpoint} HTTP/1.1\r\n"
    request += "Host: example.com\r\n"

    # Add some common headers
    request += (
        "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/96.0.4664.110\r\n"
    )

    # Add the malicious header if that's our attack vector
    if attack_type == 3:
        request += f"{header_name}: {header_value}\r\n"

    # Add a large number of headers for header-based DoS
    if random.random() < 0.3:
        for i in range(random.randint(50, 100)):
            request += f"X-Custom-Header-{i}: value{i}\r\n"

    if method in ["POST", "PUT", "PATCH"]:
        content_type = (
            "application/json"
            if random.random() < 0.7
            else "application/x-www-form-urlencoded"
        )
        request += f"Content-Type: {content_type}\r\n"

        # Create malicious body if that's our attack vector
        if attack_type == 4:
            if content_type == "application/json":
                body = f'{{"payload": "{attack_pattern}"}}'
            else:
                body = f"payload={attack_pattern}"
        else:
            # Create an unusually large body for DoS
            if content_type == "application/json":
                body = "{"
                for i in range(random.randint(100, 200)):
                    if i > 0:
                        body += ", "
                    body += f'"field{i}": "value{i}"'
                body += "}"
            else:
                body_parts = []
                for i in range(random.randint(100, 200)):
                    body_parts.append(f"field{i}=value{i}")
                body = "&".join(body_parts)

        request += f"Content-Length: {len(body)}\r\n"
        request += f"\r\n{body}"
    else:
        request += "\r\n"

    return request


def generate_mixed_dataset(
    attack_patterns: List[str],
    num_normal: int = 2500,
    num_malicious: int = 2500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a mixed dataset of normal and malicious requests.

    Args:
        attack_patterns: List of DoS attack patterns
        num_normal: Number of normal samples to generate
        num_malicious: Number of malicious samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing the mixed dataset
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Generate normal requests
    normal_requests = []
    for _ in range(num_normal):
        request = generate_normal_request()
        normal_requests.append({"request": request, "label": 0})

    # Generate malicious requests
    malicious_requests = []
    for _ in range(num_malicious):
        request = generate_malicious_request(attack_patterns)
        malicious_requests.append({"request": request, "label": 1})

    # Combine and shuffle the dataset
    dataset = pd.DataFrame(normal_requests + malicious_requests)
    dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    num_samples = len(dataset)
    logger.info(
        f"Generated DoS dataset with {num_normal} normal and {num_malicious} malicious samples, total: {num_samples}"
    )

    return dataset
