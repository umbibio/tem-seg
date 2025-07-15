"""Configuration presets for TEM-Seg.

This module provides predefined configuration presets for different organelles
and common use cases, allowing users to quickly get started with
optimized configurations.
"""

from tem_analysis_pipeline.configuration.config import (
    OrganelleType,
    PredictionConfig,
    TrainingConfig,
    WorkflowConfig,
)
from tem_analysis_pipeline.model.unet import UNetConfig


def create_mitochondria_preset(name: str = "mitochondria") -> WorkflowConfig:
    """Create a configuration preset optimized for mitochondria segmentation.

    Args:
        name: Name of the workflow configuration.

    Returns:
        WorkflowConfig optimized for mitochondria segmentation.
    """
    model_config = UNetConfig(
        input_shape=(512, 512, 1),
        filters=[64, 128, 256, 512, 1024],
        dropout_rate=0.4,
        activation="relu",
        final_activation="sigmoid",
        kernel_size=(3, 3),
        pool_size=(2, 2),
        depth=5,
        use_batch_norm=True,
        use_attention=False,
    )

    training_config = TrainingConfig(
        batch_size=8,
        epochs=200,
        learning_rate=1e-4,
        loss_function="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "iou", "precision", "recall"],
        augmentation=True,
        early_stopping=True,
        early_stopping_patience=15,
        validation_split=0.15,
        class_weights={0: 1.0, 1: 2.0},
    )

    prediction_config = PredictionConfig(
        threshold=0.5,
        overlap=128,
        batch_size=8,
        post_processing=True,
        min_object_size=100,
        fill_holes=True,
    )

    return WorkflowConfig(
        name=name,
        organelle_type=OrganelleType.MITOCHONDRIA,
        description="Optimized configuration for mitochondria segmentation",
        version="1.0.0",
        model=model_config,
        training=training_config,
        prediction=prediction_config,
    )


def create_nucleus_preset(name: str = "nucleus") -> WorkflowConfig:
    """Create a configuration preset optimized for nucleus segmentation.

    Args:
        name: Name of the workflow configuration.

    Returns:
        WorkflowConfig optimized for nucleus segmentation.
    """
    model_config = UNetConfig(
        input_shape=(768, 768, 1),
        filters=[64, 128, 256, 512],
        dropout_rate=0.3,
        activation="relu",
        final_activation="sigmoid",
        kernel_size=(3, 3),
        pool_size=(2, 2),
        depth=4,
        use_batch_norm=True,
        use_attention=False,
    )

    training_config = TrainingConfig(
        batch_size=4,
        epochs=150,
        learning_rate=5e-5,
        loss_function="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "iou", "precision", "recall"],
        augmentation=True,
        early_stopping=True,
        early_stopping_patience=20,
        validation_split=0.2,
        class_weights={0: 1.0, 1: 3.0},  # Nuclei are usually larger objects
    )

    prediction_config = PredictionConfig(
        threshold=0.6,  # Higher threshold for more confident predictions
        overlap=192,  # Larger overlap for large structures
        batch_size=4,
        post_processing=True,
        min_object_size=500,  # Larger minimum size for nuclei
        fill_holes=True,
    )

    return WorkflowConfig(
        name=name,
        organelle_type=OrganelleType.NUCLEUS,
        description="Optimized configuration for nucleus segmentation",
        version="1.0.0",
        model=model_config,
        training=training_config,
        prediction=prediction_config,
    )


def create_er_preset(name: str = "endoplasmic_reticulum") -> WorkflowConfig:
    """Create a configuration preset optimized for endoplasmic reticulum segmentation.

    Args:
        name: Name of the workflow configuration.

    Returns:
        WorkflowConfig optimized for endoplasmic reticulum segmentation.
    """
    model_config = UNetConfig(
        input_shape=(512, 512, 1),
        filters=[32, 64, 128, 256, 512],  # Start with fewer filters for fine structures
        dropout_rate=0.2,
        activation="relu",
        final_activation="sigmoid",
        kernel_size=(3, 3),
        pool_size=(2, 2),
        depth=5,
        use_batch_norm=True,
        use_attention=False,
    )

    training_config = TrainingConfig(
        batch_size=8,
        epochs=300,  # More epochs for complex structures
        learning_rate=1e-4,
        loss_function="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", "iou", "precision", "recall"],
        augmentation=True,
        early_stopping=True,
        early_stopping_patience=25,  # More patience for complex structures
        validation_split=0.15,
        class_weights={0: 1.0, 1: 4.0},  # ER is often thin and sparse
    )

    prediction_config = PredictionConfig(
        threshold=0.4,  # Lower threshold to capture fine structures
        overlap=128,
        batch_size=8,
        post_processing=True,
        min_object_size=30,  # Smaller minimum size for ER fragments
        fill_holes=False,  # ER naturally has holes
    )

    return WorkflowConfig(
        name=name,
        organelle_type=OrganelleType.ENDOPLASMIC_RETICULUM,
        description="Optimized configuration for endoplasmic reticulum segmentation",
        version="1.0.0",
        model=model_config,
        training=training_config,
        prediction=prediction_config,
    )


# Dictionary of preset creation functions by organelle type
PRESET_CREATORS = {
    OrganelleType.MITOCHONDRIA: create_mitochondria_preset,
    OrganelleType.NUCLEUS: create_nucleus_preset,
    OrganelleType.ENDOPLASMIC_RETICULUM: create_er_preset,
}


def get_preset_for_organelle(
    organelle_type: OrganelleType, name: str = None
) -> WorkflowConfig:
    """Get a configuration preset for a specific organelle type.

    Args:
        organelle_type: The organelle type to get a preset for.
        name: Optional custom name for the configuration.
            If not provided, the organelle name will be used.

    Returns:
        WorkflowConfig optimized for the specified organelle.

    Raises:
        ValueError: If no preset is available for the specified organelle type.
    """
    creator = PRESET_CREATORS.get(organelle_type)
    if not creator:
        raise ValueError(f"No preset available for organelle type: {organelle_type}")

    if name is None:
        name = organelle_type.name.lower()

    return creator(name)
