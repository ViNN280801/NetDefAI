import yaml
from pathlib import Path
from src.logger import Logger
from typing import Any, Dict, Optional, Union
from src.logger.logger_settings import config_loader_logger


class ConfigLoadError(Exception):
    """Base exception for configuration loading errors."""


class ConfigFileEmptyError(ConfigLoadError):
    """Raised when the configuration file is empty."""


class ConfigKeyNotFoundError(ConfigLoadError):
    """Raised when a required key is not found in the configuration."""


class ConfigValidationError(ConfigLoadError):
    """Raised when the configuration fails validation."""


class YamlConfigLoader:
    """
    A generic YAML configuration loader that supports arbitrary nesting levels and provides robust error handling.

    This class is designed to load YAML configuration files, validate their contents, and provide access to nested
    configuration values using dot notation (e.g., "a.b.c"). It follows SOLID principles and includes private
    methods for internal validation and error checking.

    Attributes:
        config_path (str): Path to the YAML configuration file.
        config_data (Dict[str, Any]): Loaded configuration data.
        logger (Logger): Logger instance for tracking operations and errors.
    """

    def __init__(self, config_path: Union[str, Path], logger: Optional[Logger] = None):
        """
        Initialize the YamlConfigLoader with a configuration file path.

        Args:
            config_path (str): Path to the YAML configuration file.
            logger (Optional[Logger]): Custom logger instance. Defaults to the global config_logger.
        """

        def check_file_path(filepath: Union[str, Path]) -> Union[str, Path]:
            """
            Checks if the file path exists and is a file.
            """
            from os.path import exists, isfile
            from os import access, R_OK

            if not isinstance(filepath, (str, Path)):
                raise TypeError(f"File path is not a string or Path: {type(filepath)}")
            if not filepath:
                raise ValueError("File path is empty")
            if not exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            if not isfile(filepath):
                raise FileNotFoundError(f"File is not a file: {filepath}")
            if not access(filepath, R_OK):
                raise PermissionError(f"File is not readable: {filepath}")
            return filepath

        check_file_path(config_path)

        self.config_path = config_path
        self.config_data: Dict[str, Any] = {}
        self.logger = logger or config_loader_logger

        self.__load_config()

    def __load_config(self) -> None:
        """
        Private method to load and parse the YAML configuration file.

        Raises:
            ConfigFileEmptyError: If the file is empty or contains no valid YAML data.
            ConfigLoadError: If there is an error parsing the YAML content.
        """
        self.logger.debug(f"Loading configuration from {self.config_path}...")
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            if not config_data:
                raise ConfigFileEmptyError(
                    f"Configuration file {self.config_path} is empty"
                )

            if not isinstance(config_data, dict):
                raise ConfigLoadError(
                    f"Configuration must be a dictionary, got {type(config_data)}"
                )

            self.__validate_loaded_data(config_data)
            self.config_data = config_data
            self.logger.info(
                f"Successfully loaded configuration from {self.config_path}"
            )

        except yaml.YAMLError as e:
            msg = f"Failed to parse YAML from {self.config_path}: {str(e)}"
            self.logger.error(msg)
            raise ConfigLoadError(msg)
        except Exception as e:
            msg = f"Unexpected error loading {self.config_path}: {str(e)}"
            self.logger.error(msg)
            raise ConfigLoadError(msg)

    def __validate_loaded_data(self, config_data: Any) -> None:
        """
        Private method to validate the loaded configuration data.

        Args:
            config_data (Any): The data loaded from the YAML file.

        Raises:
            ConfigFileEmptyError: If the configuration data is empty or None.
        """
        if not config_data:
            msg = f"Configuration file {self.config_path} is empty"
            self.logger.error(msg)
            raise ConfigFileEmptyError(msg)

    def __navigate_nested_keys(self, keys: list[str], data: Dict[str, Any]) -> Any:
        """
        Private method to navigate through nested dictionary keys.

        Args:
            keys (list[str]): List of keys representing the path (e.g., ["a", "b", "c"] for "a.b.c").
            data (Dict[str, Any]): The dictionary to navigate.

        Returns:
            Any: The value at the specified key path.

        Raises:
            ConfigKeyNotFoundError: If any key in the path is not found or the path traverses a non-dictionary value.
        """
        current = data
        for key in keys:
            try:
                current = current[key]
            except (KeyError, TypeError):
                msg = (
                    f"Key path '{'.'.join(keys)}' not found at '{key}' in configuration"
                )
                self.logger.error(msg)
                raise ConfigKeyNotFoundError(msg)
        return current

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieve a value from the configuration by key, supporting arbitrary nesting.

        Args:
            key (str): The key to look up in the configuration (e.g., "a.b.c" for nested access).
            default (Optional[Any]): Default value to return if the key is not found. If None, raises an exception.

        Returns:
            Any: The value associated with the key.

        Raises:
            ConfigKeyNotFoundError: If the key is not found and no default is provided.
        """
        self.logger.debug(f"Retrieving key '{key}' from configuration")
        keys = key.split(".")

        try:
            value = self.__navigate_nested_keys(keys, self.config_data)
            self.logger.debug(f"Successfully retrieved value for key '{key}': {value}")
            return value
        except ConfigKeyNotFoundError:
            if default is not None:
                self.logger.warning(
                    f"Key '{key}' not found, returning default: {default}"
                )
                return default
            raise

    def get_required(self, key: str) -> Any:
        """
        Retrieve a required value from the configuration by key, supporting arbitrary nesting.

        Args:
            key (str): The key to look up in the configuration (e.g., "a.b.c" for nested access).

        Returns:
            Any: The value associated with the key.

        Raises:
            ConfigKeyNotFoundError: If the key is not found.
        """
        return self.get(key)

    def validate_keys(self, required_keys: list[str]) -> None:
        """
        Validate that all specified keys exist in the configuration, supporting arbitrary nesting.

        Args:
            required_keys (list[str]): List of keys that must be present (e.g., ["a.b.c", "x.y"]).

        Raises:
            ConfigValidationError: If any required key is missing.
        """
        self.logger.debug(f"Validating required keys: {required_keys}")
        missing_keys = []

        for key in required_keys:
            try:
                self.get_required(key)
            except ConfigKeyNotFoundError:
                missing_keys.append(key)

        if missing_keys:
            msg = f"Missing required keys in configuration: {missing_keys}"
            self.logger.error(msg)
            raise ConfigValidationError(msg)

        self.logger.info("All required keys validated successfully")

    def get_nested_dict(self, key: str) -> Dict[str, Any]:
        """
        Retrieve a nested dictionary from the configuration by key, supporting arbitrary nesting.

        Args:
            key (str): The key to a nested dictionary (e.g., "a.b.c" for nested access).

        Returns:
            Dict[str, Any]: The nested dictionary.

        Raises:
            ConfigKeyNotFoundError: If the key is not found.
            ConfigValidationError: If the value at the key is not a dictionary.
        """
        value = self.get_required(key)
        if not isinstance(value, dict):
            msg = f"Value at '{key}' is not a dictionary: {type(value)}"
            self.logger.error(msg)
            raise ConfigValidationError(msg)
        return value
