"""
Configuration management utilities.

Functions for loading and merging configuration files.
"""

from pathlib import Path
from typing import Dict, Any, Union, Optional
import json


def load_config(config_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Args:
        config_file: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file contains invalid JSON
    """
    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file {config_file}: {e}")

    return config


def merge_configs(default_config: Dict[str, Any],
                 user_config: Optional[Dict[str, Any]] = None,
                 **kwargs) -> Dict[str, Any]:
    """
    Merge configuration dictionaries with priority order.

    Priority (highest to lowest):
    1. kwargs (command line arguments)
    2. user_config (user-provided config file)
    3. default_config (script defaults)

    Args:
        default_config: Default configuration
        user_config: User configuration (optional)
        **kwargs: Override parameters

    Returns:
        Merged configuration dictionary
    """
    # Start with defaults
    merged = default_config.copy()

    # Apply user config if provided
    if user_config:
        merged.update(user_config)

    # Apply command line overrides
    merged.update(kwargs)

    return merged


def validate_config_keys(config: Dict[str, Any], required_keys: list) -> None:
    """
    Validate that configuration contains required keys.

    Args:
        config: Configuration dictionary
        required_keys: List of required key names

    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")


def get_nested_config(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.

    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "model.temperature")
        default: Default value if key not found

    Returns:
        Configuration value or default

    Example:
        >>> config = {"model": {"temperature": 37.0}}
        >>> get_nested_config(config, "model.temperature")
        37.0
    """
    keys = key_path.split('.')
    value = config

    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default