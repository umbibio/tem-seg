"""Model registry and factory functions.

This module provides a registry of available models and functions to create models
based on their configurations.
"""

from typing import Dict, Tuple, Type

from .base import BaseModel, BaseModelConfig
from .unet import UNetConfig, UNetModel

# Registry mapping architecture names to their config and model classes
MODEL_REGISTRY: Dict[str, Tuple[Type[BaseModelConfig], Type[BaseModel]]] = {
    "unet": (UNetConfig, UNetModel),
    # Add other architectures as they are implemented:
    # "deeplabv3": (DeepLabV3Config, DeepLabV3Model),
    # "segformer": (SegFormerConfig, SegFormerModel),
}


def get_model_config_class(architecture: str) -> Type[BaseModelConfig]:
    """Get the configuration class for a model architecture.

    Args:
        architecture: Name of the architecture.

    Returns:
        The configuration class for the architecture.

    Raises:
        ValueError: If the architecture is not registered.
    """
    if architecture not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model architecture: {architecture}. "
            f"Available architectures: {available}"
        )

    return MODEL_REGISTRY[architecture][0]


def create_model_from_config(config: BaseModelConfig) -> BaseModel:
    """Create a model instance from a configuration.

    Args:
        config: Model configuration.

    Returns:
        An instance of the appropriate model class.

    Raises:
        ValueError: If the model architecture is not registered.
    """
    architecture = config.architecture

    if architecture not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model architecture: {architecture}. "
            f"Available architectures: {available}"
        )

    model_class = MODEL_REGISTRY[architecture][1]
    return model_class(config)
