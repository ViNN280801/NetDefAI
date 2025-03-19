from pathlib import Path
from typing import Union


class UtilsError(Exception):
    """Base exception for utils errors"""


class UtilsTypeError(UtilsError):
    """Raised when a value is not of the expected type"""


class UtilsValueError(UtilsError):
    """Raised when a value is not valid"""


def check_file_path(filepath: Union[str, Path]) -> Union[str, Path]:
    """
    Checks if the file path exists and is a file.
    """
    from os.path import exists, isfile
    from os import access, R_OK

    if not isinstance(filepath, (str, Path)):
        raise UtilsTypeError(f"File path is not a string or Path: {type(filepath)}")
    if not filepath:
        raise UtilsValueError("File path is empty")
    if not exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    if not isfile(filepath):
        raise FileNotFoundError(f"File is not a file: {filepath}")
    if not access(filepath, R_OK):
        raise PermissionError(f"File is not readable: {filepath}")
    return filepath


def truncate_long_string(value: str, max_length: int = 20) -> str:
    """
    Truncates long strings for logging purposes.
    If string is longer than max_length, shows first and last 5 characters.
    """
    if not isinstance(value, str):
        raise UtilsTypeError(f"Value is not a string: {type(value)}")
    if not isinstance(max_length, int):
        raise UtilsTypeError(f"Max length is not an integer: {type(max_length)}")
    if max_length <= 0:
        raise UtilsValueError(f"Max length is not positive: {max_length}")
    if not value:
        raise UtilsValueError("Value is empty, nothing to truncate")

    # If the value is a valid UUID4 or the length is less than the max length, return it as is
    if len(value) <= max_length or is_uuid4(value):
        return value
    return f"{value[:max_length//2]}...{value[-max_length//2:]}"


def is_uuid4(value: str) -> bool:
    """Checks if the value is a valid UUID4"""
    if not isinstance(value, str):
        raise UtilsTypeError(f"Value '{value}' is not a string: {type(value)}")
    if not value:
        raise UtilsValueError(f"Value '{value}' is empty, nothing to check")

    try:
        from uuid import UUID

        return bool(UUID(value, version=4))
    except ValueError:
        return False
