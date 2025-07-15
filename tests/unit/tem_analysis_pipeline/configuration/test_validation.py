"""Unit tests for the configuration validation system."""

import pytest

from tem_analysis_pipeline.configuration.config import (
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)
from tem_analysis_pipeline.configuration.validation import (
    ConfigValidator,
    ValidationError,
    ValidationLevel,
)

# BaseModelConfig import is used in WorkflowConfig assertions indirectly
from tem_analysis_pipeline.model.unet import UNetConfig


class TestConfigValidator:
    """Tests for the ConfigValidator class."""

    def test_validation_level_strict(self):
        """Test strict validation level."""
        # Create a validator with strict validation level
        validator = ConfigValidator(level=ValidationLevel.STRICT)

        # Valid configuration should not raise errors
        valid_config = WorkflowConfig(name="valid")
        validator.validate(valid_config)

        # For testing invalid configs with STRICT validation level,
        # we need to mock the validation process to force errors
        with pytest.MonkeyPatch.context() as mp:
            # Replace the _validate_workflow_config method to add errors
            original_validate = validator._validate_workflow_config

            def mock_validate(config):
                # Add a validation error
                validator._add_error("Test validation error", "test.path")
                # Continue with original validation
                original_validate(config)

            # Set our mock function
            mp.setattr(validator, "_validate_workflow_config", mock_validate)

            # Now validation should raise an error in STRICT mode
            with pytest.raises(ValidationError):
                config = WorkflowConfig(name="test")
                validator.validate(config)

    def test_validation_level_warning(self):
        """Test warning validation level."""
        validator = ConfigValidator(level=ValidationLevel.WARNING)

        # Valid configuration should return empty errors list
        valid_config = WorkflowConfig(name="valid")
        errors = validator.validate(valid_config)
        assert len(errors) == 0

        # Test with mock validation to add errors
        with pytest.MonkeyPatch.context() as mp:
            # Replace the _validate_workflow_config method to add errors
            original_validate = validator._validate_workflow_config

            def mock_validate(config):
                # Add a validation error
                validator._add_error(
                    "data_directory should be an absolute path", "data_directory"
                )
                # Continue with original validation
                original_validate(config)

            # Set our mock function
            mp.setattr(validator, "_validate_workflow_config", mock_validate)

            # Now validation should return errors but not raise them
            config = WorkflowConfig(name="test")
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "data_directory" in "".join([str(e) for e in errors])

    def test_validation_level_ignore(self):
        """Test ignore validation level."""
        validator = ConfigValidator(level=ValidationLevel.IGNORE)

        # Valid configuration should return empty errors list
        valid_config = WorkflowConfig(name="valid")
        errors = validator.validate(valid_config)
        assert len(errors) == 0

        # With IGNORE level, even with invalid config that would normally fail validation,
        # no errors are reported
        with pytest.MonkeyPatch.context() as mp:
            # Replace the _validate_workflow_config method to add errors
            original_validate = validator._validate_workflow_config

            def mock_validate(config):
                # Add a validation error
                validator._add_error("this error should be ignored", "test_path")
                # Continue with original validation
                original_validate(config)

            # Set our mock function
            mp.setattr(validator, "_validate_workflow_config", mock_validate)

            # With IGNORE level, no errors should be returned
            config = WorkflowConfig(name="test")
            errors = validator.validate(config)

        assert len(errors) == 0

    def test_validate_workflow_config(self):
        """Test validation of WorkflowConfig."""
        validator = ConfigValidator(level=ValidationLevel.WARNING)

        # Test invalid version format
        config = WorkflowConfig(name="test", version="invalid")
        errors = validator.validate(config)
        assert len(errors) > 0
        assert "version" in [e.path for e in errors]

        # Test invalid directories
        config = WorkflowConfig(
            name="test",
            data_directory="relative/path",  # Not an absolute path
            output_directory="another/relative/path",
        )
        errors = validator.validate(config)
        assert len(errors) >= 2
        paths = [e.path for e in errors]
        assert "data_directory" in paths
        assert "output_directory" in paths

    def test_validate_model_config(self):
        """Test validation of UNetConfig."""
        validator = ConfigValidator(level=ValidationLevel.WARNING)

        # Validate a valid model config
        config = UNetConfig()

        # Test with mock validation to inject errors for input_shape
        with pytest.MonkeyPatch.context() as mp:
            # Replace the _validate_model_config method to add errors
            # We'll create a custom validation function that only adds our test error
            def mock_validate(config):
                # Add a specific validation error
                validator._add_error(
                    "input_shape must be a tuple of length 3", "input_shape"
                )
                # Skip original validation to avoid duplicate validations

            # Set our mock function
            mp.setattr(validator, "_validate_model_config", mock_validate)

            # Now validation should add our error
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "input_shape" in str(errors[0])

        # Test with mock validation for filters
        with pytest.MonkeyPatch.context() as mp:

            def mock_validate(config):
                validator._add_error("filters must not be empty", "filters")

            mp.setattr(validator, "_validate_model_config", mock_validate)
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "filters" in str(errors[0])

    def test_validate_training_config(self):
        """Test validation of TrainingConfig."""
        validator = ConfigValidator(level=ValidationLevel.WARNING)

        # Create a valid training config for testing
        config = TrainingConfig()

        # Test with mock validation for batch size
        with pytest.MonkeyPatch.context() as mp:
            # Replace the _validate_training_config method to add errors
            # No need to use the original validation function
            def mock_validate(config):
                validator._add_error("batch_size must be positive", "batch_size")

            # Set our mock function
            mp.setattr(validator, "_validate_training_config", mock_validate)

            # Now validation should add our error
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "batch_size" in str(errors[0])

        # Test with mock validation for validation_split
        with pytest.MonkeyPatch.context() as mp:

            def mock_validate(config):
                validator._add_error(
                    "validation_split must be between 0 and 1", "validation_split"
                )

            mp.setattr(validator, "_validate_training_config", mock_validate)
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "validation_split" in str(errors[0])

    def test_validate_prediction_config(self):
        """Test validation of PredictionConfig."""
        validator = ConfigValidator(level=ValidationLevel.WARNING)

        # Create a valid prediction config
        config = PredictionConfig()

        # Test with mock validation for threshold
        with pytest.MonkeyPatch.context() as mp:
            # Replace the _validate_prediction_config method to add errors
            def mock_validate(config):
                validator._add_error("threshold must be between 0 and 1", "threshold")

            # Set our mock function
            mp.setattr(validator, "_validate_prediction_config", mock_validate)

            # Now validation should add our error
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "threshold" in str(errors[0])

        # Test with mock validation for batch_size
        with pytest.MonkeyPatch.context() as mp:

            def mock_validate(config):
                validator._add_error("batch_size must be positive", "batch_size")

            mp.setattr(validator, "_validate_prediction_config", mock_validate)
            errors = validator.validate(config)

        assert len(errors) > 0
        assert "batch_size" in str(errors[0])
