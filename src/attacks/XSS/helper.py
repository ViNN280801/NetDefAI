#!/usr/bin/env python3
"""
Helper module for XSS attack dataset generation.

This module provides functions to generate datasets for XSS attack detection.
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


# Sample HTML tags for normal payloads
normal_TAGS = [
    "div",
    "span",
    "p",
    "h1",
    "h2",
    "h3",
    "ul",
    "ol",
    "li",
    "table",
    "tr",
    "td",
    "a",
    "img",
    "form",
    "input",
    "button",
]

# Sample attributes for normal payloads
normal_ATTRIBUTES = {
    "class": [
        "container",
        "header",
        "footer",
        "btn",
        "main",
        "content",
        "sidebar",
        "nav",
    ],
    "id": ["main", "header", "footer", "sidebar", "content", "nav"],
    "style": ["color: blue;", "margin: 10px;", "padding: 5px;", "font-weight: bold;"],
    "href": ["https://example.com", "#section1", "/page2.html", "/contact"],
    "src": ["/images/logo.png", "https://example.com/image.jpg"],
    "type": ["text", "button", "submit", "password"],
    "alt": ["Logo", "Profile picture", "Banner image"],
}


def generate_normal_payload() -> str:
    """
    Generate a normal HTML/JavaScript payload.

    Returns:
        A string representing a normal HTML/JavaScript payload
    """
    payload_type = random.choice(["text", "html", "html_with_js"])

    if payload_type == "text":
        # Generate plain text
        words = [
            "Hello",
            "World",
            "Welcome",
            "to",
            "our",
            "website",
            "Please",
            "login",
            "This",
            "is",
            "a",
            "test",
            "message",
            "Have",
            "a",
            "nice",
            "day",
        ]

        num_words = random.randint(3, 15)
        payload = " ".join(random.choices(words, k=num_words))

        # Sometimes add punctuation
        if random.random() < 0.5:
            payload += random.choice([".", "!", "?"])

    elif payload_type == "html":
        # Generate an HTML snippet without JavaScript
        tag = random.choice(normal_TAGS)

        # Some tags are self-closing
        self_closing = tag in ["img", "input"]

        # Add attributes
        attributes = []
        num_attrs = random.randint(0, 3)

        for _ in range(num_attrs):
            attr_name = random.choice(list(normal_ATTRIBUTES.keys()))
            # Only add appropriate attributes for the tag
            if (attr_name == "src" and tag != "img") or (
                attr_name == "href" and tag != "a"
            ):
                continue

            attr_value = random.choice(normal_ATTRIBUTES[attr_name])
            attributes.append(f'{attr_name}="{attr_value}"')

        # Build the tag
        if attributes:
            tag_open = f"<{tag} {' '.join(attributes)}"
        else:
            tag_open = f"<{tag}"

        if self_closing:
            payload = f"{tag_open} />"
        else:
            # Add content
            content_options = [
                "Hello World",
                "Welcome to our website",
                "Please login to continue",
                "This is a sample text",
                f"<{random.choice(normal_TAGS)}>"
                f"{random.choice(['Text', 'Content', 'Info'])}"
                f"</{random.choice(normal_TAGS)}>",
            ]
            content = random.choice(content_options)

            payload = f"{tag_open}>{content}</{tag}>"

    else:  # html_with_js
        # Generate normal JavaScript
        js_options = [
            "document.getElementById('demo').innerHTML = 'Hello World';",
            "function showMessage() { alert('Welcome!'); }",
            "var x = document.getElementById('myInput').value;",
            "document.querySelector('.container').style.display = 'block';",
            "console.log('User logged in');",
            "window.onload = function() { init(); };",
        ]

        js_code = random.choice(js_options)

        # Add to script tag
        payload = f"<script>{js_code}</script>"

        # Sometimes embed in an event handler
        if random.random() < 0.3:
            tag = random.choice(["button", "a", "div"])
            event = random.choice(["onclick", "onmouseover", "onchange"])
            js_function = random.choice(["showMessage()", "validate()", "updateUI()"])

            payload = f'<{tag} {event}="{js_function}">Click me</{tag}>'

    return payload


def generate_malicious_payload(attack_patterns: List[str]) -> str:
    """
    Generate a malicious XSS payload using attack patterns.

    Args:
        attack_patterns: List of XSS attack patterns

    Returns:
        A string representing a malicious XSS payload
    """
    # Choose a random attack pattern
    attack_pattern = random.choice(attack_patterns)

    # There are several ways to present an XSS payload
    attack_type = random.randint(1, 4)

    if attack_type == 1:
        # Use the raw attack pattern
        payload = attack_pattern
    elif attack_type == 2:
        # Embed in some normal-looking text
        words = ["Hello", "Welcome", "Please", "This", "Click", "Enter", "Submit"]
        prefix = random.choice(words)
        payload = f"{prefix} {attack_pattern}"
    elif attack_type == 3:
        # URL encode parts of the attack
        # This is a simplified encoding for demonstration
        payload = attack_pattern.replace("<", "%3C").replace(">", "%3E")
    else:
        # Mix with normal HTML
        tag = random.choice(normal_TAGS)
        attr = random.choice(list(normal_ATTRIBUTES.keys()))

        if random.random() < 0.5:
            # Malicious attribute
            payload = f'<{tag} {attr}="{attack_pattern}">Content</{tag}>'
        else:
            # Malicious content
            payload = f"<{tag}>{attack_pattern}</{tag}>"

    return payload


def generate_mixed_dataset(
    attack_patterns: List[str],
    num_normal: int = 2500,
    num_malicious: int = 2500,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a mixed dataset of normal and malicious XSS payloads.

    Args:
        attack_patterns: List of XSS attack patterns
        num_normal: Number of normal samples to generate
        num_malicious: Number of malicious samples to generate
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame containing the mixed dataset
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Generate normal payloads
    normal_payloads = []
    for _ in range(num_normal):
        payload = generate_normal_payload()
        normal_payloads.append({"payload": payload, "label": 0})

    # Generate malicious payloads
    malicious_payloads = []
    for _ in range(num_malicious):
        payload = generate_malicious_payload(attack_patterns)
        malicious_payloads.append({"payload": payload, "label": 1})

    # Combine and shuffle the dataset
    dataset = pd.DataFrame(normal_payloads + malicious_payloads)
    dataset = dataset.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info(
        f"Generated XSS dataset with {num_normal} normal and {num_malicious} malicious samples"
    )

    return dataset
