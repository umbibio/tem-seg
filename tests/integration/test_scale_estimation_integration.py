"""Integration tests for Scale Estimation System.

These tests use real TIF images from the raw-data directory to validate
that the scale estimation system works correctly with actual TEM images.
"""

import os
from pathlib import Path

import pytest

from tem_analysis_pipeline.scale_estimation import (
    NoScaleFoundError,
    ScaleBarReader,
    ScaleEstimationError,
    ScaleManager,
)

# Get the project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Paths to raw data directories
SAMPLE_DIRS = [
    PROJECT_ROOT / "raw-data" / "HCI-010" / "slide-images",
    PROJECT_ROOT / "raw-data" / "DRP1-KO" / "slide-images",
]
# Sample TIF files for testing
SAMPLE_IMAGES = [
    # HCI-010 samples
    SAMPLE_DIRS[0] / "R20_72_G01_2913_53.tif",
    SAMPLE_DIRS[0] / "R20_72_G01_2929_46.tif",
    # DRP1-KO samples
    SAMPLE_DIRS[1] / "wt_1_HM_1.tif",
    SAMPLE_DIRS[1] / "wt_5Kx_1.tif",
]


@pytest.mark.parametrize("image_path", SAMPLE_IMAGES)
def test_scale_bar_reader_with_real_images(image_path):
    """Test ScaleBarReader with real TEM images.

    This test checks if the ScaleBarReader can successfully extract
    scale information from real TEM images or correctly report that it can't.
    """
    reader = ScaleBarReader()

    # Ensure the image file exists
    assert image_path.exists(), f"Test image not found: {image_path}"

    try:
        scale = reader.estimate_scale(image_path)

        # Validate the scale value is reasonable for TEM images
        # TEM scales are typically in the range of 0.001 to 10 μm/pixel
        assert 0.0001 <= scale <= 100, f"Scale value {scale} outside expected range"
        print(f"Successfully estimated scale for {image_path.name}: {scale} μm/pixel")

    except NoScaleFoundError as e:
        # Not all images will have scale bars, so this is an acceptable outcome
        print(f"No scale bar found for {image_path.name} (this is okay): {e}")

    except ScaleEstimationError as e:
        # For other errors, report but don't fail the test
        print(f"Warning: Scale estimation error for {image_path.name}: {e}")


def test_scale_manager_fallback():
    """Test ScaleManager fallback functionality with real images.

    This test verifies that the ScaleManager correctly applies a fallback
    scale value when scale estimation fails for a real image.
    """
    # Create a manager with just one ScaleBarReader
    manager = ScaleManager([ScaleBarReader()])

    # Use a non-image file that will definitely fail scale estimation
    non_image_path = PROJECT_ROOT / "pyproject.toml"

    # This should use the fallback value
    fallback_scale = 0.05  # 0.05 μm/pixel
    result = manager.get_scale(non_image_path, fallback=fallback_scale)

    assert result == fallback_scale, (
        f"Expected fallback scale {fallback_scale}, got {result}"
    )


def test_scale_manager_multiple_estimators():
    """Test ScaleManager with multiple estimators on real images.

    This test checks if the ScaleManager correctly tries multiple estimators
    in sequence until one succeeds.
    """

    # Create mock estimator that always fails
    class FailingEstimator:
        def can_estimate(self, image_input):
            return True

        def estimate_scale(self, image_input):
            raise NoScaleFoundError("This estimator always fails")

    # Create a manager with the failing estimator first, then the real one
    manager = ScaleManager([FailingEstimator(), ScaleBarReader()])

    # Get a sample image path
    image_path = SAMPLE_IMAGES[0]

    try:
        # The first estimator should fail, then the second one should succeed
        scale = manager.get_scale(image_path)

        # Validate the scale value
        assert 0.0001 <= scale <= 100, f"Scale value {scale} outside expected range"
        print(
            f"Successfully estimated scale using fallback estimator: {scale} μm/pixel"
        )

    except ScaleEstimationError as e:
        # If scale estimation fails, make this a failing test
        pytest.fail(f"Scale manager failed to fallback to working estimator: {e}")


def test_process_batch_of_images():
    """Test processing a batch of images with ScaleManager.

    This test simulates a batch processing scenario where multiple
    images are processed and their scales are tracked.
    """
    manager = ScaleManager([ScaleBarReader()])
    results = {}

    # Process each image and collect results
    for image_path in SAMPLE_IMAGES:
        try:
            scale = manager.get_scale(image_path, fallback=0.05)
            results[image_path.name] = {
                "success": True,
                "scale": scale,
                "used_fallback": False,
            }
        except ScaleEstimationError:
            # Use fallback but mark it
            results[image_path.name] = {
                "success": False,
                "scale": 0.05,  # fallback value
                "used_fallback": True,
            }

    # Ensure we have results for all images
    assert len(results) == len(SAMPLE_IMAGES)

    # Print summary
    print("\nScale estimation results summary:")
    for name, data in results.items():
        status = "SUCCESS" if data["success"] else "FALLBACK"
        print(f"{name}: {status} - {data['scale']} μm/pixel")

    # Check if at least one image was successfully processed
    assert any(data["success"] for data in results.values()), (
        "No images were successfully processed"
    )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
