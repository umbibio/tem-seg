from pathlib import Path
from typing import Any, Dict, List

import toml

__all__ = [
    "load_config_files",
]


def load_config_files(config_files: List[str | Path]) -> Dict[str, Any]:
    """Load and merge TOML configuration files."""
    config = {}

    for file_path in config_files:
        file_path = Path(file_path)
        if file_path.exists():
            try:
                file_config = toml.load(file_path)
                config = _deep_merge(config, file_config)
            except Exception as e:
                print(f"    ❌ Error loading: {e}")

    return config


def _deep_merge(
    base_dict: Dict[str, Any], override_dict: Dict[str, Any]
) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base_dict.copy()

    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
