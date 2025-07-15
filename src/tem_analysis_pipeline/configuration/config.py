"""Configuration dataclasses for TEM-Seg.

This module provides dataclasses that represent various configuration options
for the TEM-Seg application. These classes enable type-safe configuration
with validation, serialization to/from YAML, and versioning support.
"""

import dataclasses
import hashlib
import inspect
import json
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

import yaml

# Define a type variable for the model config to avoid circular imports
ModelConfigType = TypeVar("ModelConfigType")

# Type variable for the configuration class
T = TypeVar("T", bound="ConfigurationBase")


class OrganelleType(str, Enum):
    """Types of organelles that can be segmented.

    Inherits from str to make it compatible with Typer CLI.
    """

    MITOCHONDRIA = "mitochondria"
    NUCLEUS = "nucleus"
    ENDOPLASMIC_RETICULUM = "er"  # CLI-friendly version
    GOLGI_APPARATUS = "golgi_apparatus"
    LYSOSOME = "lysosome"
    PEROXISOME = "peroxisome"
    CUSTOM = "custom"


@dataclass(frozen=True)
class ConfigurationBase:
    """Base class for all configuration objects.

    Provides common functionality for serialization, deserialization,
    validation, and versioning.
    """

    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """Create a configuration from a dictionary.

        Args:
            config_dict: Dictionary with configuration values.

        Returns:
            Configuration instance.
        """
        # Get all fields defined in the dataclass
        fields_dict = {field.name: field for field in dataclasses.fields(cls)}

        # Create a new dict with only the fields that are in the dataclass
        filtered_dict = {}

        for key, value in config_dict.items():
            if key in fields_dict:
                field_type = fields_dict[key].type

                # Handle special cases for conversion
                if hasattr(field_type, "__origin__") and field_type.__origin__ is tuple:
                    # Convert lists to tuples if the field type is a tuple
                    if isinstance(value, list):
                        value = tuple(value)

                # Handle Enum conversion
                if inspect.isclass(field_type) and issubclass(field_type, Enum):
                    for enum_val in field_type:
                        if enum_val.value == value:
                            value = enum_val
                            break

                # Handle the model configuration specially
                elif key == "model" and isinstance(value, dict):
                    # If the model dict has an 'architecture' key, use it to create the right config class
                    # Import here to avoid circular imports
                    from ..model.registry import get_model_config_class

                    architecture = value.get("architecture", "unet")  # Default to UNet
                    try:
                        model_config_class = get_model_config_class(architecture)
                        value = model_config_class.from_dict(value)
                    except ValueError:
                        # Fall back to UNet if architecture not found
                        from ..model.unet.config import UNetConfig

                        value = UNetConfig.from_dict(value)

                # Handle other nested dataclasses
                elif is_dataclass(field_type) and isinstance(value, dict):
                    value = field_type.from_dict(value)

                filtered_dict[key] = value

        # Create and return the instance
        return cls(**filtered_dict)

    @classmethod
    def from_yaml(cls: Type[T], yaml_str: str) -> T:
        """Create a configuration object from a YAML string.

        Args:
            yaml_str: YAML string containing configuration.

        Returns:
            A new configuration object.

        Raises:
            ValueError: If the YAML string contains invalid values.
        """
        try:
            config_dict = yaml.safe_load(yaml_str)
            if not isinstance(config_dict, dict):
                raise ValueError("YAML must contain a dictionary")
            return cls.from_dict(config_dict)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {str(e)}") from e

    @classmethod
    def from_file(cls: Type[T], file_path: Union[str, Path]) -> T:
        """Create a configuration object from a YAML file.

        Args:
            file_path: Path to a YAML file containing configuration.

        Returns:
            A new configuration object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file contains invalid YAML or configuration values.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r") as f:
                yaml_str = f.read()
            return cls.from_yaml(yaml_str)
        except (IOError, ValueError) as e:
            raise ValueError(
                f"Failed to load configuration from {path}: {str(e)}"
            ) from e

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        result = {}
        for key, value in self.__dict__.items():
            if key.startswith("_"):
                continue

            # Handle nested configuration objects
            if isinstance(value, ConfigurationBase):
                value = value.to_dict()

            # Convert enum values to their string representation
            elif isinstance(value, Enum):
                value = value.value

            # Convert tuples to lists for better serialization
            elif isinstance(value, tuple):
                value = list(value)

            result[key] = value
        return result

    def to_yaml(self) -> str:
        """Convert the configuration to a YAML string.

        Returns:
            YAML string representation of the configuration.
        """
        # Use safe_dump to avoid custom tags like !!python/tuple
        # to_dict method will convert tuples to lists for safe serialization
        return yaml.safe_dump(self.to_dict(), default_flow_style=False)

    def save(self, file_path: Union[str, Path]) -> None:
        """Save the configuration to a YAML file.

        Args:
            file_path: Path to save the configuration to.

        Raises:
            IOError: If the file cannot be written.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            f.write(self.to_yaml())

    def get_hash(self) -> str:
        """Get a hash of the configuration.

        This can be used for versioning or identifying configuration changes.

        Returns:
            Hash string unique to this configuration.
        """
        # to_dict already handles enum conversion properly
        config_dict = self.to_dict()
        # Convert to JSON and hash
        config_json = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_json.encode()).hexdigest()

    def __eq__(self, other: Any) -> bool:
        """Check if two configurations are equal.

        Args:
            other: Another configuration object to compare with.

        Returns:
            True if the configurations are equal, False otherwise.
        """
        if not isinstance(other, ConfigurationBase):
            return NotImplemented
        return self.to_dict() == other.to_dict()


@dataclass(frozen=True)
class TrainingConfig(ConfigurationBase):
    """Configuration for model training."""

    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 1e-4
    loss_function: str = "binary_crossentropy"
    optimizer: str = "adam"
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "iou"])
    augmentation: bool = True
    early_stopping: bool = True
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    class_weights: Optional[Dict[int, float]] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Basic validations that should always happen regardless of validation level
        pass


@dataclass(frozen=True)
class PredictionConfig(ConfigurationBase):
    """Configuration for model prediction."""

    threshold: float = 0.5
    overlap: int = 64
    batch_size: int = 16
    post_processing: bool = True
    min_object_size: int = 50
    fill_holes: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        # Only perform critical validations
        pass


@dataclass(frozen=True)
class WorkflowConfig(ConfigurationBase):
    """Configuration for a segmentation workflow."""

    name: str
    organelle_type: OrganelleType = OrganelleType.MITOCHONDRIA
    description: str = ""
    version: str = "1.0.0"
    model: Any = field(
        default_factory=lambda: None
    )  # Will be set to a model config instance
    training: TrainingConfig = field(default_factory=TrainingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    data_directory: Optional[str] = None
    output_directory: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not self.name:
            raise ValueError("name must not be empty")

        # If model is None, initialize with default UNetConfig
        if self.model is None:
            # Import here to avoid circular imports
            from ..model.unet.config import UNetConfig

            object.__setattr__(self, "model", UNetConfig())
