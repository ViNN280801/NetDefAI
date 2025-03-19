# /src/attacks/SQLInjections/__init__.py

"""
SQL Injection attack detection modules.
"""

from .helper import (
    generate_normal_query,
    generate_malicious_query,
    generate_mixed_dataset,
)
from .sql_query_analyzer import (
    is_sql_query,
    is_query_allowed,
    load_model,
    classify_query,
    explain_query,
)

__all__ = [
    "generate_normal_query",
    "generate_malicious_query",
    "generate_mixed_dataset",
    "is_sql_query",
    "is_query_allowed",
    "load_model",
    "classify_query",
    "explain_query",
]
