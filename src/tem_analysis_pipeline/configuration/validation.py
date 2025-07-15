"""Configuration validation utilities.

This module provides tools for validating configuration objects against
schemas and ensuring configuration compatibility.
"""

import re
from enum import Enum
from pathlib import Path
from typing import List, Optional

from tem_analysis_pipeline.configuration.config import (
    ConfigurationBase,
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)
from tem_analysis_pipeline.model.base import BaseModelConfig


class ValidationLevel(Enum):
    """Level of validation strictness."""

    STRICT = "strict"  # Fail on any validation error
    WARNING = "warning"  # Log warnings but don't fail
    IGNORE = "ignore"  # Skip validation entirely


class ConfigurationError(Exception):
    """Base class for configuration-related exceptions."""

    pass


class ValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""

    def __init__(self, message: str, path: Optional[str] = None) -> None:
        """Initialize a validation error.

        Args:
            message: Error message.
            path: Path to the configuration element that failed validation.
        """
        if path:
            message = f"{path}: {message}"
        super().__init__(message)
        self.path = path


class ConfigValidator:
    """Validates configuration objects against schemas and constraints."""

    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT) -> None:
        """Initialize a configuration validator.

        Args:
            level: Validation strictness level.
        """
        self.level = level
        self._errors: List[ValidationError] = []

    def validate(self, config: ConfigurationBase) -> List[ValidationError]:
        """Validate a configuration object.

        Args:
            config: Configuration object to validate.

        Returns:
            List of validation errors, empty if validation succeeded.

        Raises:
            ValidationError: If validation fails and level is STRICT.
        """
        self._errors = []

        # Skip validation if level is IGNORE
        if self.level == ValidationLevel.IGNORE:
            return self._errors

        # Dispatch to appropriate validation method based on config type
        if isinstance(config, WorkflowConfig):
            self._validate_workflow_config(config)
        elif isinstance(config, BaseModelConfig):
            self._validate_model_config(config)
        elif isinstance(config, TrainingConfig):
            self._validate_training_config(config)
        elif isinstance(config, PredictionConfig):
            self._validate_prediction_config(config)

        # Raise the first error if level is STRICT
        if self.level == ValidationLevel.STRICT and self._errors:
            raise self._errors[0]

        return self._errors

    def _add_error(self, message: str, path: Optional[str] = None) -> None:
        """Add a validation error.

        Args:
            message: Error message.
            path: Path to the configuration element that failed validation.
        """
        self._errors.append(ValidationError(message, path))

    def _validate_workflow_config(self, config: WorkflowConfig) -> None:
        """Validate a workflow configuration.

        Args:
            config: Workflow configuration to validate.
        """
        # Validate name
        if not config.name or not config.name.strip():
            self._add_error("Workflow name cannot be empty", "name")

        # Validate version format
        version_pattern = r"^\d+\.\d+\.\d+$"
        if not re.match(version_pattern, config.version):
            self._add_error(
                f"Version '{config.version}' does not follow semantic versioning (X.Y.Z)",
                "version",
            )

        # Validate directories if provided
        if config.data_directory:
            data_dir = Path(config.data_directory)
            if not data_dir.is_absolute():
                self._add_error(
                    f"Data directory '{config.data_directory}' is not an absolute path",
                    "data_directory",
                )

        if config.output_directory:
            output_dir = Path(config.output_directory)
            if not output_dir.is_absolute():
                self._add_error(
                    f"Output directory '{config.output_directory}' is not an absolute path",
                    "output_directory",
                )

        # Validate nested configurations
        self._validate_model_config(config.model)
        self._validate_training_config(config.training)
        self._validate_prediction_config(config.prediction)

    def _validate_model_config(self, config: BaseModelConfig) -> None:
        """Validate a model configuration.

        Args:
            config: Model configuration to validate.
        """
        # Validate input shape
        if len(config.input_shape) != 3:
            self._add_error(
                f"Input shape must have 3 dimensions, got {len(config.input_shape)}",
                "model.input_shape",
            )

        for dim in config.input_shape:
            if dim <= 0:
                self._add_error(
                    f"Input shape dimensions must be positive, got {config.input_shape}",
                    "model.input_shape",
                )

        # Validate filters
        if not config.filters:
            self._add_error("Filters list cannot be empty", "model.filters")

        for f in config.filters:
            if f <= 0:
                self._add_error(
                    f"Filter values must be positive, got {f}", "model.filters"
                )

        # Validate dropout rate
        if not 0 <= config.dropout_rate <= 1:
            self._add_error(
                f"Dropout rate must be between 0 and 1, got {config.dropout_rate}",
                "model.dropout_rate",
            )

        # Validate kernel and pool sizes
        for size, name in [
            (config.kernel_size, "kernel_size"),
            (config.pool_size, "pool_size"),
        ]:
            if len(size) != 2:
                self._add_error(
                    f"{name} must have 2 dimensions, got {len(size)}", f"model.{name}"
                )
            for dim in size:
                if dim <= 0:
                    self._add_error(
                        f"{name} dimensions must be positive, got {size}",
                        f"model.{name}",
                    )

    def _validate_training_config(self, config: TrainingConfig) -> None:
        """Validate a training configuration.

        Args:
            config: Training configuration to validate.
        """
        # Validate batch size
        if config.batch_size <= 0:
            self._add_error(
                f"Batch size must be positive, got {config.batch_size}",
                "training.batch_size",
            )

        # Validate epochs
        if config.epochs <= 0:
            self._add_error(
                f"Epochs must be positive, got {config.epochs}", "training.epochs"
            )

        # Validate learning rate
        if config.learning_rate <= 0:
            self._add_error(
                f"Learning rate must be positive, got {config.learning_rate}",
                "training.learning_rate",
            )

        # Validate validation split
        if not 0 <= config.validation_split < 1:
            self._add_error(
                f"Validation split must be between 0 and 1, got {config.validation_split}",
                "training.validation_split",
            )

        # Validate early stopping patience
        if config.early_stopping and config.early_stopping_patience <= 0:
            self._add_error(
                f"Early stopping patience must be positive, got {config.early_stopping_patience}",
                "training.early_stopping_patience",
            )

        # Validate metrics
        if not config.metrics:
            self._add_error("Metrics list cannot be empty", "training.metrics")

        # Validate class weights if present
        if config.class_weights:
            for class_idx, weight in config.class_weights.items():
                if not isinstance(class_idx, int) or class_idx < 0:
                    self._add_error(
                        f"Class indices must be non-negative integers, got {class_idx}",
                        "training.class_weights",
                    )
                if weight <= 0:
                    self._add_error(
                        f"Class weights must be positive, got {weight} for class {class_idx}",
                        "training.class_weights",
                    )

    def _validate_prediction_config(self, config: PredictionConfig) -> None:
        """Validate a prediction configuration.

        Args:
            config: Prediction configuration to validate.
        """
        # Validate threshold
        if not 0 <= config.threshold <= 1:
            self._add_error(
                f"Threshold must be between 0 and 1, got {config.threshold}",
                "prediction.threshold",
            )

        # Validate overlap
        if config.overlap < 0:
            self._add_error(
                f"Overlap must be non-negative, got {config.overlap}",
                "prediction.overlap",
            )

        # Validate batch size
        if config.batch_size <= 0:
            self._add_error(
                f"Batch size must be positive, got {config.batch_size}",
                "prediction.batch_size",
            )

        # Validate min object size
        if config.min_object_size < 0:
            self._add_error(
                f"Minimum object size must be non-negative, got {config.min_object_size}",
                "prediction.min_object_size",
            )
