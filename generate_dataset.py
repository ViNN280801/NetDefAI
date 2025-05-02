#!/usr/bin/env python3
"""
Unified dataset generator for web attack detection.

This script generates training and testing datasets for different types of web attacks
by combining normal requests with attack patterns from the patterns directory.
"""

import os
import sys
import random
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.logger.logger_settings import dataset_generator_logger as logger

# Add parent directory to path so we can import modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# Constants
ATTACK_TYPES = ["dos", "path_traversal", "sql_injection", "xss"]
PATTERNS_DIR = os.path.join(current_dir, "patterns")
OUTPUT_DIR = os.path.join(current_dir, "datasets")
ATTACKS_DIR = os.path.join(current_dir, "attacks")


def random_case(s: str) -> str:
    return "".join(c.upper() if c.isalpha() and random.random() < 0.5 else c for c in s)


def insert_whitespace(s: str, p: float = 0.05) -> str:
    return "".join(c + (" " if random.random() < p else "") for c in s)


def fuzz_percent_encode(s: str, p: float = 0.1) -> str:
    out = []
    for c in s:
        if random.random() < p and c not in "/:?=& \r\n":
            out.append(f"%{ord(c):02X}")
        else:
            out.append(c)
    return "".join(out)


def load_attack_patterns(attack_type: str) -> List[str]:
    """
    Load attack patterns from the patterns directory.

    Args:
        attack_type: Type of attack (dos, path_traversal, sql_injection, xss)

    Returns:
        List of attack patterns
    """
    pattern_file_map = {
        "dos": "dos.txt",
        "path_traversal": "traversal.txt",
        "sql_injection": "sql_injections.txt",
        "xss": "xss.txt",
    }

    file_path = os.path.join(
        PATTERNS_DIR, pattern_file_map.get(attack_type, f"{attack_type}.txt")
    )

    if not os.path.exists(file_path):
        logger.error(f"Pattern file not found: {file_path}")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        patterns = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]

    logger.info(f"Loaded {len(patterns)} attack patterns for {attack_type}")
    return patterns


def get_attack_helper(attack_type: str):
    """
    Dynamically import the appropriate attack helper module.

    Args:
        attack_type: Type of attack (dos, path_traversal, sql_injection, xss)

    Returns:
        Attack helper module
    """
    # Standardize the attack type name for import
    module_name_map = {
        "dos": "DoS",
        "path_traversal": "PathTraversal",
        "sql_injection": "SQLInjections",
        "xss": "XSS",
    }

    module_name = module_name_map.get(attack_type, attack_type)

    try:
        # Try to import the helper module
        module_path = f"attacks.{module_name}.helper"
        helper_module = __import__(module_path, fromlist=[""])
        logger.info(f"Successfully imported helper for {attack_type}")
        return helper_module
    except ImportError as e:
        logger.error(f"Failed to import helper for {attack_type}: {e}")
        raise ImportError(f"No helper module found for attack type: {attack_type}")


def generate_dataset(
    attack_type: str,
    num_samples: int = 5000,
    malicious_ratio: float = 0.5,
    output_file: Optional[str] = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a dataset for the specified attack type.

    Args:
        attack_type: Type of attack (dos, path_traversal, sql_injection, xss)
        num_samples: Total number of samples to generate
        malicious_ratio: Ratio of malicious samples to normal samples
        output_file: Path to save the dataset
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing the generated dataset
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load attack patterns
    attack_patterns = load_attack_patterns(attack_type)

    if not attack_patterns:
        logger.error(f"No attack patterns found for {attack_type}")
        return pd.DataFrame()

    # Get the appropriate attack helper
    try:
        helper = get_attack_helper(attack_type)
    except ImportError:
        logger.error(f"Failed to get helper for {attack_type}")
        return pd.DataFrame()

    # Calculate the number of malicious and normal samples
    num_malicious = int(num_samples * malicious_ratio)
    num_normal = num_samples - num_malicious

    logger.info(
        f"Generating dataset for {attack_type}: {num_malicious} malicious, {num_normal} normal"
    )

    # Generate dataset using the helper
    try:
        # The helper should implement a generate_mixed_dataset function
        dataset = helper.generate_mixed_dataset(
            attack_patterns=attack_patterns,
            num_normal=num_normal,
            num_malicious=num_malicious,
            random_seed=random_seed,
        )

        dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(
            drop=True
        )
    except Exception as e:
        logger.error(f"Error generating dataset with helper: {e}")
        return pd.DataFrame()

    def obfuscate_text(s: str) -> str:
        return insert_whitespace(fuzz_percent_encode(random_case(s), p=0.15), p=0.05)

    for col in ("request", "path", "query", "payload"):
        if col in dataset.columns:
            dataset[col] = dataset[col].apply(obfuscate_text)

    # Ensure the output directory exists
    if output_file:
        output_path = output_file
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, f"{attack_type}_dataset.csv")

    # Save the dataset
    try:
        dataset.to_csv(output_path, index=False)
        logger.info(f"Dataset saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save dataset: {e}")

    return dataset


def generate_all_datasets(
    num_samples: int = 5000, malicious_ratio: float = 0.5, random_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Generate datasets for all attack types.

    Args:
        num_samples: Total number of samples to generate per attack type
        malicious_ratio: Ratio of malicious samples to normal samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping attack types to their datasets
    """
    datasets = {}

    for attack_type in ATTACK_TYPES:
        logger.info(f"Generating dataset for {attack_type}")
        dataset = generate_dataset(
            attack_type=attack_type,
            num_samples=num_samples,
            malicious_ratio=malicious_ratio,
            random_seed=random_seed,
        )

        if not dataset.empty:
            datasets[attack_type] = dataset

    return datasets


def main():
    """Main function for the dataset generator."""
    parser = argparse.ArgumentParser(
        description="Generate datasets for web attack detection"
    )
    parser.add_argument(
        "--attack-type",
        "-a",
        choices=ATTACK_TYPES + ["all"],
        default="all",
        help="Type of attack to generate dataset for (default: all)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=5000,
        help="Number of samples to generate (default: 5000)",
    )
    parser.add_argument(
        "--malicious-ratio",
        "-r",
        type=float,
        default=0.5,
        help="Ratio of malicious samples (default: 0.5)",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        type=str,
        help="Output file path (default: datasets/{attack_type}_dataset.csv)",
    )
    parser.add_argument(
        "--random-seed",
        "-s",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    if args.attack_type == "all":
        datasets = generate_all_datasets(
            num_samples=args.num_samples,
            malicious_ratio=args.malicious_ratio,
            random_seed=args.random_seed,
        )
        logger.info(f"Generated datasets for {len(datasets)} attack types")
    else:
        dataset = generate_dataset(
            attack_type=args.attack_type,
            num_samples=args.num_samples,
            malicious_ratio=args.malicious_ratio,
            output_file=args.output_file,
            random_seed=args.random_seed,
        )

        if not dataset.empty:
            logger.info(
                f"Generated dataset for {args.attack_type}: {len(dataset)} samples"
            )
        else:
            logger.error(f"Failed to generate dataset for {args.attack_type}")


if __name__ == "__main__":
    main()
