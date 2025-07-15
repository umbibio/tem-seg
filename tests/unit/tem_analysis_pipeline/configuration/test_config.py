"""Unit tests for the configuration dataclasses."""

import os
import tempfile

import pytest
import yaml

from tem_analysis_pipeline.configuration.config import (
    OrganelleType,
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)
from tem_analysis_pipeline.model.base import BaseModelConfig
from tem_analysis_pipeline.model.unet import UNetConfig
from tem_analysis_pipeline.configuration.validation import (
    ConfigValidator,
    ValidationLevel,
)


class TestConfigurationBase:
    """Tests for the ConfigurationBase class."""

    def test_from_dict(self):
        """Test creating a configuration from a dictionary."""
        config_dict = {"name": "test_config", "description": "Test configuration"}
        config = WorkflowConfig.from_dict(config_dict)
        assert config.name == "test_config"
        assert config.description == "Test configuration"
        # Should use default values for fields not in the dictionary
        assert config.version == "1.0.0"

    def test_from_dict_ignores_unknown_fields(self):
        """Test that from_dict ignores unknown fields."""
        config_dict = {
            "name": "test_config",
            "unknown_field": "should be ignored",
        }
        config = WorkflowConfig.from_dict(config_dict)
        assert config.name == "test_config"
        # The unknown field should not raise an error

    def test_from_dict_invalid_values(self):
        """Test from_dict with invalid values."""
        config_dict = {"name": ""}  # Empty name is invalid
        with pytest.raises(ValueError):
            WorkflowConfig.from_dict(config_dict)

    def test_to_dict(self):
        """Test converting a configuration to a dictionary."""
        config = WorkflowConfig(name="test_config")
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test_config"
        assert config_dict["version"] == "1.0.0"
        assert "model" in config_dict
        assert "training" in config_dict
        assert "prediction" in config_dict

    def test_from_yaml(self):
        """Test creating a configuration from a YAML string."""
        yaml_str = """
        name: yaml_config
        description: Created from YAML
        """
        config = WorkflowConfig.from_yaml(yaml_str)
        assert config.name == "yaml_config"
        assert config.description == "Created from YAML"

    def test_from_yaml_invalid(self):
        """Test from_yaml with invalid YAML."""
        yaml_str = "{"  # Invalid YAML
        with pytest.raises(ValueError):
            WorkflowConfig.from_yaml(yaml_str)

    def test_to_yaml(self):
        """Test converting a configuration to a YAML string."""
        config = WorkflowConfig(name="test_config")
        yaml_str = config.to_yaml()
        assert isinstance(yaml_str, str)

        # Parse the YAML string and verify its contents
        parsed = yaml.safe_load(yaml_str)
        assert parsed["name"] == "test_config"
        assert parsed["organelle_type"] == "mitochondria"

    def test_save_and_load(self):
        """Test saving and loading a configuration to/from a file."""
        config = WorkflowConfig(
            name="save_test",
            description="Testing save and load",
        )

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Save the configuration
            config.save(tmp_path)

            # Verify the file exists and has content
            assert os.path.exists(tmp_path)
            with open(tmp_path, "r") as f:
                content = f.read()
                assert "name: save_test" in content
                assert "description: Testing save and load" in content

            # Load the configuration - our fixed from_dict method should handle enums correctly now
            loaded_config = WorkflowConfig.from_file(tmp_path)

            # Check that the loaded config matches the original
            assert loaded_config.name == config.name
            assert loaded_config.description == config.description
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_from_file_not_found(self):
        """Test from_file with a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            WorkflowConfig.from_file("/nonexistent/file.yaml")

    def test_get_hash(self):
        """Test getting a hash of a configuration."""
        config1 = WorkflowConfig(name="config1")
        config2 = WorkflowConfig(name="config2")
        config1_copy = WorkflowConfig(name="config1")

        # Generate hashes
        hash1 = config1.get_hash()
        hash2 = config2.get_hash()
        hash1_copy = config1_copy.get_hash()

        # Same configurations should have the same hash
        assert hash1 == hash1_copy
        # Different configurations should have different hashes
        assert hash1 != hash2
        # Hash should be a string
        assert isinstance(hash1, str)
        # Hash should be a valid SHA-256 hex digest (64 chars)
        assert len(hash1) == 64

    def test_equality(self):
        """Test configuration equality comparison."""
        config1 = WorkflowConfig(name="config1")
        config2 = WorkflowConfig(name="config2")
        config1_copy = WorkflowConfig(name="config1")

        assert config1 == config1_copy
        assert config1 != config2
        assert config1 != "not a config"  # Different types


class TestUNetConfig:
    """Tests for the UNetConfig class."""

    def test_default_values(self):
        """Test the default values of UNetConfig."""
        config = UNetConfig()
        assert config.architecture == "unet"
        assert config.input_shape == (256, 256, 1)
        assert len(config.filters) == 4
        assert config.kernel_size == (3, 3)
        assert config.pool_size == (2, 2)
        assert config.depth == 4
        assert config.use_attention is False

    def test_custom_values(self):
        """Test creating a UNetConfig with custom values."""
        config = UNetConfig(
            input_shape=(512, 512, 3),
            filters=[16, 32, 64, 128],
            kernel_size=(5, 5),
            pool_size=(3, 3),
            depth=4,
            use_attention=True,
        )
        assert config.architecture == "unet"  # Architecture is fixed for UNetConfig
        assert config.input_shape == (512, 512, 3)
        assert config.filters == [16, 32, 64, 128]
        assert config.kernel_size == (5, 5)
        assert config.pool_size == (3, 3)
        assert config.depth == 4
        assert config.use_attention is True

    def test_validation(self):
        """Test validation of UNetConfig values."""
        # Create a valid model config
        config = UNetConfig()

        # Test with ConfigValidator
        validator = ConfigValidator(level=ValidationLevel.WARNING)
        errors = validator.validate(config)
        assert len(errors) == 0


class TestTrainingConfig:
    """Tests for the TrainingConfig class."""

    def test_default_values(self):
        """Test the default values of TrainingConfig."""
        config = TrainingConfig()
        assert config.batch_size == 16
        assert config.epochs == 100
        assert config.learning_rate == 1e-4
        assert "accuracy" in config.metrics

    def test_custom_values(self):
        """Test creating a TrainingConfig with custom values."""
        config = TrainingConfig(
            batch_size=32,
            epochs=50,
            learning_rate=1e-3,
            metrics=["dice_coefficient", "iou"],
            class_weights={0: 1.0, 1: 3.0},
        )
        assert config.batch_size == 32
        assert config.epochs == 50
        assert config.learning_rate == 1e-3
        assert config.metrics == ["dice_coefficient", "iou"]
        assert config.class_weights == {0: 1.0, 1: 3.0}

    def test_validation(self):
        """Test validation of TrainingConfig values."""
        # Create a valid training config
        config = TrainingConfig()

        # Test with ConfigValidator
        validator = ConfigValidator(level=ValidationLevel.WARNING)
        errors = validator.validate(config)
        assert len(errors) == 0


class TestPredictionConfig:
    """Tests for the PredictionConfig class."""

    def test_default_values(self):
        """Test the default values of PredictionConfig."""
        config = PredictionConfig()
        assert config.threshold == 0.5
        assert config.overlap == 64
        assert config.batch_size == 16
        assert config.post_processing is True
        assert config.min_object_size == 50
        assert config.fill_holes is True

    def test_custom_values(self):
        """Test creating a PredictionConfig with custom values."""
        config = PredictionConfig(
            threshold=0.7,
            overlap=128,
            batch_size=8,
            post_processing=False,
            min_object_size=100,
            fill_holes=False,
        )
        assert config.threshold == 0.7
        assert config.overlap == 128
        assert config.batch_size == 8
        assert config.post_processing is False
        assert config.min_object_size == 100
        assert config.fill_holes is False

    def test_validation(self):
        """Test validation of PredictionConfig values."""
        # Create a valid prediction config
        config = PredictionConfig()

        # Test with ConfigValidator
        validator = ConfigValidator(level=ValidationLevel.WARNING)
        errors = validator.validate(config)
        assert len(errors) == 0

        # Since we're now doing external validation, the __post_init__ validations are minimal
        # No need to test constructor validation anymore


class TestWorkflowConfig:
    """Tests for the WorkflowConfig class."""

    def test_default_values(self):
        """Test the default values of WorkflowConfig."""
        config = WorkflowConfig(name="test")
        assert config.name == "test"
        assert config.organelle_type == OrganelleType.MITOCHONDRIA
        assert config.version == "1.0.0"
        assert isinstance(config.model, BaseModelConfig)
        # Default model config should be a UNetConfig
        assert isinstance(config.model, UNetConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.prediction, PredictionConfig)

    def test_custom_values(self):
        """Test creating a WorkflowConfig with custom values."""
        model_config = UNetConfig()
        training_config = TrainingConfig(batch_size=32)
        prediction_config = PredictionConfig(threshold=0.7)

        config = WorkflowConfig(
            name="custom_workflow",
            organelle_type=OrganelleType.NUCLEUS,
            description="Custom workflow configuration",
            version="2.0.0",
            model=model_config,
            training=training_config,
            prediction=prediction_config,
            data_directory="/data",
            output_directory="/output",
        )

        assert config.name == "custom_workflow"
        assert config.organelle_type == OrganelleType.NUCLEUS
        assert config.description == "Custom workflow configuration"
        assert config.version == "2.0.0"
        assert config.model is model_config
        assert config.model.architecture == "unet"
        assert config.training is training_config
        assert config.training.batch_size == 32
        assert config.prediction is prediction_config
        assert config.prediction.threshold == 0.7
        assert config.data_directory == "/data"
        assert config.output_directory == "/output"

    def test_validation(self):
        """Test validation of WorkflowConfig values."""
        # Invalid name (empty)
        with pytest.raises(ValueError):
            WorkflowConfig(name="")
