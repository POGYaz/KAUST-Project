"""
Configuration management utilities for the recommendation system.

This module provides functions to load, validate, and manage configuration
files in YAML format, with support for environment variable substitution
and configuration merging.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        Dictionary containing the loaded configuration.
        
    Raises:
        ConfigError: If the configuration file cannot be loaded or parsed.
        FileNotFoundError: If the configuration file does not exist.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ConfigError(f"Empty configuration file: {config_path}")
            
        return config
        
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error loading configuration file {config_path}: {e}")


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save.
        config_path: Path where to save the configuration file.
        
    Raises:
        ConfigError: If the configuration cannot be saved.
    """
    config_path = Path(config_path)
    
    try:
        # Create parent directories if they don't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
            
    except Exception as e:
        raise ConfigError(f"Error saving configuration file {config_path}: {e}")


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later configurations override earlier ones. Nested dictionaries
    are merged recursively.
    
    Args:
        *configs: Variable number of configuration dictionaries to merge.
        
    Returns:
        Merged configuration dictionary.
    """
    if not configs:
        return {}
    
    result = configs[0].copy()
    
    for config in configs[1:]:
        result = _deep_merge(result, config)
    
    return result


def _deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.
    
    Args:
        base: Base dictionary to merge into.
        update: Dictionary with updates to apply.
        
    Returns:
        Merged dictionary.
    """
    result = base.copy()
    
    for key, value in update.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Substitute environment variables in configuration values.
    
    Supports ${VAR_NAME} and ${VAR_NAME:default_value} syntax.
    
    Args:
        config: Configuration dictionary with potential environment variables.
        
    Returns:
        Configuration dictionary with environment variables substituted.
    """
    def _substitute_value(value: Any) -> Any:
        if isinstance(value, str):
            return _substitute_string(value)
        elif isinstance(value, dict):
            return {k: _substitute_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [_substitute_value(item) for item in value]
        else:
            return value
    
    return _substitute_value(config)


def _substitute_string(text: str) -> str:
    """
    Substitute environment variables in a string.
    
    Args:
        text: String potentially containing environment variables.
        
    Returns:
        String with environment variables substituted.
    """
    import re
    
    def replace_var(match):
        var_expr = match.group(1)
        if ':' in var_expr:
            var_name, default_value = var_expr.split(':', 1)
            return os.environ.get(var_name.strip(), default_value)
        else:
            var_name = var_expr.strip()
            return os.environ.get(var_name, match.group(0))
    
    # Match ${VAR_NAME} or ${VAR_NAME:default}
    pattern = r'\$\{([^}]+)\}'
    return re.sub(pattern, replace_var, text)


def validate_config(config: Dict[str, Any], required_keys: list[str]) -> None:
    """
    Validate that required keys are present in configuration.
    
    Args:
        config: Configuration dictionary to validate.
        required_keys: List of required key paths (use '.' for nested keys).
        
    Raises:
        ConfigError: If any required keys are missing.
    """
    missing_keys = []
    
    for key_path in required_keys:
        if not _has_nested_key(config, key_path):
            missing_keys.append(key_path)
    
    if missing_keys:
        raise ConfigError(f"Missing required configuration keys: {missing_keys}")


def _has_nested_key(config: Dict[str, Any], key_path: str) -> bool:
    """
    Check if a nested key exists in configuration.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path (e.g., 'model.learning_rate').
        
    Returns:
        True if the key path exists, False otherwise.
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return False
        current = current[key]
    
    return True


def get_config_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get a value from nested configuration using dot notation.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated key path (e.g., 'model.learning_rate').
        default: Default value if key path is not found.
        
    Returns:
        Configuration value or default if not found.
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    
    return current
