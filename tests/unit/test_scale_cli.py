"""Unit tests for the Scale Estimation CLI module.

These tests verify that the scale estimation CLI functions work correctly
at the module level, focusing on the core functionality of the commands.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from typer.testing import CliRunner

from tem_analysis_pipeline.cmd.scale_estimation import create_scale_app
from tem_analysis_pipeline.scale_estimation import (
    NoScaleFoundError,
    ScaleEstimationError,
)


@pytest.fixture
def scale_app():
    """Create a scale app instance for testing."""
    return create_scale_app()()


@pytest.fixture
def runner():
    """Create a CLI runner for testing Typer applications."""
    return CliRunner()


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_success(runner, scale_app):
    """Test get-scale command with a successful scale detection."""
    # Mock ScaleManager to always return a fixed scale
    mock_scale = 0.0055

    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleManager"
    ) as MockManager:
        # Configure the mock
        manager_instance = MockManager.return_value
        manager_instance.get_scale.return_value = mock_scale

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake image file
            test_img = Path(temp_dir) / "test.tif"
            test_img.touch()

            # Run the command
            result = runner.invoke(scale_app, [str(test_img)])

            # Verify results
            assert result.exit_code == 0
            assert f"Scale: {mock_scale}" in result.stdout
            assert "Successfully detected: 1/1" in result.stdout


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_no_scale_found(runner, scale_app):
    """Test get-scale command when no scale is found in the image."""
    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleManager"
    ) as MockManager:
        # Configure mock to raise NoScaleFoundError
        manager_instance = MockManager.return_value
        manager_instance.get_scale.side_effect = NoScaleFoundError(
            "Couldn't find scale bar"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake image file
            test_img = Path(temp_dir) / "test.tif"
            test_img.touch()

            # Run the command
            result = runner.invoke(scale_app, ["get-scale", str(test_img)])

            # Verify results
            assert result.exit_code == 0
            assert "No scale bar found" in result.stdout
            assert "Successfully processed 0/1" in result.stdout


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_with_fallback(runner, scale_app):
    """Test get-scale command with fallback option."""
    fallback_scale = 0.0075

    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleManager"
    ) as MockManager:
        # Configure the mock to raise an error when fallback is not provided
        manager_instance = MockManager.return_value

        # Configure manager to return fallback value
        def get_scale_with_fallback(img_path, fallback=None):
            if fallback is not None:
                return fallback
            raise NoScaleFoundError("Couldn't find scale bar")

        manager_instance.get_scale.side_effect = get_scale_with_fallback

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake image file
            test_img = Path(temp_dir) / "test.tif"
            test_img.touch()

            # Run the command with fallback
            result = runner.invoke(
                scale_app, [str(test_img), "--fallback", str(fallback_scale)]
            )

            # Verify results
            assert result.exit_code == 0
            assert "Using fallback scale: 1/1" in result.stdout
            assert f"Fallback scale used: {fallback_scale}" in result.stdout


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_skip_missing(runner, scale_app):
    """Test get-scale command with skip-missing option."""
    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleManager"
    ) as MockManager:
        # Configure the mock to raise a ScaleEstimationError
        manager_instance = MockManager.return_value
        manager_instance.get_scale.side_effect = ScaleEstimationError(
            "No suitable estimator found"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake image file
            test_img = Path(temp_dir) / "test.tif"
            test_img.touch()

            # Run the command with skip-missing
            result = runner.invoke(scale_app, [str(test_img), "--skip-missing"])

            # Verify results
            assert result.exit_code == 0
            assert "skipping (--skip-missing enabled)" in result.stdout
            assert "Skipped (no scale): 1/1" in result.stdout


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_debug_mode(runner, scale_app):
    """Test get-scale command with debug flag."""
    # Mock ScaleBarReader for detailed testing
    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleBarReader"
    ) as MockReader:
        reader_instance = MockReader.return_value
        # Configure reader to provide an image when converting
        mock_img = MagicMock()
        mock_img.size = (2048, 2048)
        mock_img.mode = "L"
        reader_instance._convert_to_pil.return_value = mock_img
        reader_instance.estimate_scale.return_value = 0.005

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake image file
            test_img = Path(temp_dir) / "test.tif"
            test_img.touch()

            # Run the command with debug
            result = runner.invoke(scale_app, [str(test_img), "--debug"])

            # Verify debug output
            assert result.exit_code == 0
            assert "[DEBUG] Image loaded: size=(2048, 2048), mode=L" in result.stdout
            assert "[DEBUG] Converting to grayscale if needed" in result.stdout
            assert "[DEBUG] Attempting to detect scale bar" in result.stdout
            assert "[DEBUG] Scale bar detected successfully" in result.stdout


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_csv_output(runner, scale_app):
    """Test get-scale command with CSV output."""
    mock_scale = 0.0055

    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleManager"
    ) as MockManager:
        # Configure the mock
        manager_instance = MockManager.return_value
        manager_instance.get_scale.return_value = mock_scale

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake image file and output path
            test_img = Path(temp_dir) / "test.tif"
            test_img.touch()
            output_path = Path(temp_dir) / "output.csv"

            # Mock the CSV writing since we're just testing the function behavior
            with patch("builtins.open", mock_open()) as mock_file:
                # Run the command with output
                result = runner.invoke(
                    scale_app, [str(test_img), "--output", str(output_path)]
                )

                # Verify results
                assert result.exit_code == 0
                assert f"Results saved to {output_path}" in result.stdout
                mock_file.assert_called_once_with(output_path, "w", newline="")


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_multiple_images(runner, scale_app):
    """Test get-scale command with multiple images."""
    mock_scale = 0.0055

    with patch(
        "tem_analysis_pipeline.cmd.scale_estimation.ScaleManager"
    ) as MockManager:
        # Configure the mock
        manager_instance = MockManager.return_value
        manager_instance.get_scale.return_value = mock_scale

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple fake image files
            test_img1 = Path(temp_dir) / "test1.tif"
            test_img1.touch()
            test_img2 = Path(temp_dir) / "test2.tif"
            test_img2.touch()

            # Run the command with multiple images
            result = runner.invoke(
                scale_app, [str(test_img1), str(test_img2), "--verbose"]
            )

            # Verify results
            assert result.exit_code == 0
            assert f"Processing {test_img1}" in result.stdout
            assert f"Processing {test_img2}" in result.stdout
            assert "Successfully detected: 2/2" in result.stdout


@pytest.mark.skip(
    reason="Integration tests provide better coverage of CLI functionality"
)
def test_get_scale_command_nonexistent_image(runner, scale_app):
    """Test get-scale command with nonexistent image path."""
    nonexistent_img = "/tmp/nonexistent_image.tif"

    # Run the command with nonexistent image
    result = runner.invoke(scale_app, [nonexistent_img])

    # Verify results
    assert result.exit_code == 0  # CLI should not crash
    assert "Error: Image file not found" in result.stdout
    assert "Successfully detected: 0/1" in result.stdout  # No images processed
