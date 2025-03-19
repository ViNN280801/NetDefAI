# /src/attacks/XSS/__init__.py

"""
Cross-site scripting (XSS) attack detection modules.
"""

from .helper import (
    generate_normal_payload,
    generate_malicious_payload,
    generate_mixed_dataset,
)
from .xss_analyzer import (
    is_html_content,
    is_content_allowed,
    load_model,
    classify_content,
    explain_content,
)

__all__ = [
    "generate_normal_payload",
    "generate_malicious_payload",
    "generate_mixed_dataset",
    "is_html_content",
    "is_content_allowed",
    "load_model",
    "classify_content",
    "explain_content",
]
