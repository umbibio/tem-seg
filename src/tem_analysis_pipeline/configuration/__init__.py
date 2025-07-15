"""Configuration system for TEM-Seg.

This package provides a flexible and robust configuration system for
managing settings throughout the TEM-Seg application.
"""

from tem_analysis_pipeline.configuration.config import (
    ConfigurationBase,
    OrganelleType,
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)

# Import BaseModelConfig from the model module
from tem_analysis_pipeline.model.base import BaseModelConfig

__all__ = [
    "ConfigurationBase",
    "WorkflowConfig",
    "BaseModelConfig",  # Replace ModelConfig with BaseModelConfig
    "TrainingConfig",
    "PredictionConfig",
    "OrganelleType",
]
