"""Base model definitions and configurations.

This module defines the base classes for model configurations and implementations.
"""

from dataclasses import dataclass
from enum import Enum

from ..configuration.config import ConfigurationBase


class ModelFramework(Enum):
    """Supported deep learning frameworks."""

    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"


@dataclass(frozen=True)
class BaseModelConfig(ConfigurationBase):
    """Base configuration for all model architectures.

    This class defines common parameters shared by all model architectures.
    Each specific architecture should subclass this and add architecture-specific parameters.
    """

    framework: ModelFramework = ModelFramework.TENSORFLOW
    dropout_rate: float = 0.3
    activation: str = "relu"
    final_activation: str = "sigmoid"
    use_batch_norm: bool = True

    # Required method to identify the model architecture
    @property
    def architecture(self) -> str:
        """Return the architecture name.

        This must be implemented by subclasses and should return a string
        that matches an entry in the model registry.
        """
        raise NotImplementedError("Subclasses must implement this property")


class BaseModel:
    """Base class for all models.

    This class defines the interface that all model implementations must follow.
    """

    def __init__(self, config: BaseModelConfig):
        """Initialize the model with the given configuration.

        Args:
            config: Configuration for the model.
        """
        self.config = config

    def build(self):
        """Build the model architecture.

        Returns:
            The built model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")
