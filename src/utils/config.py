"""Configuration utilities with variable substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Union

import yaml


def substitute_variables(obj: Any, root_dir: str) -> Any:
    """Recursively substitute ${root_dir} variables in config objects.
    
    Args:
        obj: Configuration object (dict, list, or primitive)
        root_dir: Root directory to substitute for ${root_dir}
        
    Returns:
        Configuration with substituted variables
    """
    if isinstance(obj, dict):
        return {k: substitute_variables(v, root_dir) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [substitute_variables(item, root_dir) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("${root_dir}", root_dir)
    else:
        return obj


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load YAML config file with variable substitution.
    
    Supports ${root_dir} variable which will be replaced with the
    value from root_dir field in the config.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary with substituted variables
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    
    with open(path) as f:
        data = yaml.safe_load(f)
    
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format: {path}")
    
    # Get root_dir from config
    root_dir = data.get("root_dir")
    if root_dir:
        # Substitute ${root_dir} in all string values
        data = substitute_variables(data, root_dir)
    
    return data
