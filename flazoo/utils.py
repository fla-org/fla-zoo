import os
import json
from typing import Optional, Dict, Any, Type, Union
from transformers.configuration_utils import PretrainedConfig


def save_config_to_json(
    config: PretrainedConfig, save_dir: str, filename: Optional[str] = None
) -> str:
    """Save a configuration object to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    filename = filename or "config.json"
    file_path = os.path.join(save_dir, filename)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    return file_path


def load_config_from_json(
    config_path: str, config_class: Type[PretrainedConfig]
) -> PretrainedConfig:
    """Load a configuration from a JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    return config_class.from_dict(config_dict)


def create_default_config_json(
    config_class: Type[PretrainedConfig],
    save_dir: str,
    filename: Optional[str] = None,
    **kwargs,
) -> str:
    """Create a JSON file with default configuration values."""
    config = config_class(**kwargs)
    return save_config_to_json(config, save_dir, filename)
