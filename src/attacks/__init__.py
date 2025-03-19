# /src/attacks/__init__.py

"""
Attack detection modules for different types of web attacks.
"""

from .DoS import __all__ as dos_all
from .PathTraversal import __all__ as path_traversal_all
from .SQLInjections import __all__ as sql_injection_all
from .XSS import __all__ as xss_all

__all__ = dos_all + path_traversal_all + sql_injection_all + xss_all  # type: ignore
