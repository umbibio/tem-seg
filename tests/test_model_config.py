"""Test the new model configuration structure."""

# We're not using pytest fixtures in this test
from tem_analysis_pipeline.configuration import WorkflowConfig, OrganelleType
from tem_analysis_pipeline.model.registry import create_model_from_config
from tem_analysis_pipeline.model.unet.config import UNetConfig


def test_workflow_with_unet_config():
    """Test creating a workflow with a UNetConfig."""
    # Create a UNetConfig with custom parameters
    unet_config = UNetConfig(
        input_shape=(512, 512, 1),
        filters=[32, 64, 128, 256],
        dropout_rate=0.4,
        use_attention=True,
    )

    # Create a workflow with the UNetConfig
    workflow = WorkflowConfig(
        name="test_workflow",
        organelle_type=OrganelleType.MITOCHONDRIA,
        model=unet_config,
        description="Test workflow with UNet",
    )

    # Check that the model config is correctly set
    assert workflow.model.architecture == "unet"
    assert workflow.model.input_shape == (512, 512, 1)
    assert workflow.model.filters == [32, 64, 128, 256]
    assert workflow.model.dropout_rate == 0.4
    assert workflow.model.use_attention is True

    # Serialize and deserialize the workflow
    workflow_yaml = workflow.to_yaml()
    loaded_workflow = WorkflowConfig.from_yaml(workflow_yaml)

    # Check that the model config was correctly serialized and deserialized
    assert loaded_workflow.model.architecture == "unet"
    assert loaded_workflow.model.input_shape == (512, 512, 1)
    assert loaded_workflow.model.filters == [32, 64, 128, 256]
    assert loaded_workflow.model.dropout_rate == 0.4
    assert loaded_workflow.model.use_attention is True

    # Test that we can create a model from the config
    model = create_model_from_config(loaded_workflow.model)
    keras_model = model.build()

    # Verify that the model was built with the correct input shape
    assert keras_model.input_shape[1:] == (512, 512, 1)


def test_default_model_config():
    """Test that a default UNetConfig is created if none is provided."""
    # Create a workflow without specifying a model config
    workflow = WorkflowConfig(
        name="default_workflow", description="Test default model config"
    )

    # Check that a UNetConfig was automatically created
    assert workflow.model.architecture == "unet"
    assert workflow.model.input_shape == (256, 256, 1)  # Default value

    # Serialize and deserialize the workflow
    workflow_yaml = workflow.to_yaml()
    loaded_workflow = WorkflowConfig.from_yaml(workflow_yaml)

    # Check that the model config was correctly serialized and deserialized
    assert loaded_workflow.model.architecture == "unet"


if __name__ == "__main__":
    # Run the tests manually
    test_workflow_with_unet_config()
    test_default_model_config()
    print("All tests passed!")
