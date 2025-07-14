"""Integration tests for the command-line interface (CLI).

These tests verify that the CLI commands work correctly, including
the restructured CLI with v1 subcommand and the new scale commands.
"""

import os
import csv
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tem_analysis_pipeline.cmd import app


# Get the project root directory
PROJECT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# Paths to raw data directories
SAMPLE_DIRS = [
    PROJECT_ROOT / "raw-data" / "HCI-010" / "slide-images",
    PROJECT_ROOT / "raw-data" / "DRP1-KO" / "slide-images"
]
# Sample TIF files for testing
SAMPLE_IMAGES = [
    # HCI-010 samples (known to have scale bars)
    SAMPLE_DIRS[0] / "R20_72_G01_2913_53.tif",
    # DRP1-KO samples (some without clear scale bars)
    SAMPLE_DIRS[1] / "rwt_2_HM_1.tif",
]


@pytest.fixture
def runner():
    """Create a CLI runner for testing Typer applications."""
    return CliRunner()


def test_cli_structure(runner):
    """Test that the CLI has the expected command structure."""
    # Test root app help
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    # Check that both v1 and scale subcommands are visible in help
    assert "v1" in result.stdout
    assert "scale" in result.stdout

    # Test v1 subcommand existence
    result = runner.invoke(app, ["v1", "--help"])
    assert result.exit_code == 0
    # Check for legacy commands
    legacy_commands = ["train", "predict", "analyze", "consolidate", "preprocess"]
    for cmd in legacy_commands:
        assert cmd in result.stdout
    
    # Test scale subcommand existence
    result = runner.invoke(app, ["scale", "--help"])
    assert result.exit_code == 0
    assert "get-scale" in result.stdout


def test_scale_get_scale_basic(runner):
    """Test the get-scale command with basic functionality."""
    # Test with a sample image that has a scale bar
    sample_image = SAMPLE_IMAGES[0]
    # Skip if image doesn't exist (CI environment might not have sample data)
    if not sample_image.exists():
        pytest.skip(f"Test image not found: {sample_image}")
    
    result = runner.invoke(app, ["scale", "get-scale", str(sample_image)])
    
    assert result.exit_code == 0
    # Check if scale was detected (should contain μm/pixel)
    assert "μm/pixel" in result.stdout
    assert "Successfully detected: 1/1" in result.stdout


def test_scale_get_scale_fallback(runner):
    """Test the get-scale command with fallback option."""
    # Use the second sample image which might not have a clear scale bar
    sample_image = SAMPLE_IMAGES[1]
    # Skip if image doesn't exist
    if not sample_image.exists():
        pytest.skip(f"Test image not found: {sample_image}")
    
    fallback = 0.005
    result = runner.invoke(app, [
        "scale", "get-scale", str(sample_image), 
        "--fallback", str(fallback)
    ])
    
    assert result.exit_code == 0
    # If this image has no scale bar, we should get the fallback message
    if "Using fallback scale:" in result.stdout:
        assert f"Fallback scale used: {fallback}" in result.stdout
        assert "Using fallback scale: 1/1" in result.stdout


def test_scale_get_scale_skip_missing(runner):
    """Test the get-scale command with skip-missing option."""
    # Use the second sample image which might not have a clear scale bar
    sample_image = SAMPLE_IMAGES[1]
    # Skip if image doesn't exist
    if not sample_image.exists():
        pytest.skip(f"Test image not found: {sample_image}")
    
    result = runner.invoke(app, [
        "scale", "get-scale", str(sample_image), 
        "--skip-missing"
    ])
    
    assert result.exit_code == 0
    # Check if the skip message is present
    if "skipping (--skip-missing enabled)" in result.stdout:
        assert "Skipped (no scale): 1/1" in result.stdout


def test_scale_get_scale_csv_output(runner):
    """Test the get-scale command with CSV output."""
    # Use a sample image
    sample_image = SAMPLE_IMAGES[0]
    # Skip if image doesn't exist
    if not sample_image.exists():
        pytest.skip(f"Test image not found: {sample_image}")
    
    # Create a temporary file for CSV output
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as temp_file:
        temp_path = Path(temp_file.name)
    
    try:
        result = runner.invoke(app, [
            "scale", "get-scale", str(sample_image),
            "--output", str(temp_path)
        ])
        
        assert result.exit_code == 0
        assert f"Results saved to {temp_path}" in result.stdout
        
        # Check if CSV file was created and has the right format
        assert temp_path.exists()
        with open(temp_path, 'r') as f:
            csv_reader = csv.DictReader(f)
            rows = list(csv_reader)
            assert len(rows) == 1
            assert "image" in rows[0]
            assert "scale" in rows[0]
            assert "status" in rows[0]
    finally:
        # Clean up the temporary file
        if temp_path.exists():
            os.unlink(temp_path)


def test_scale_get_scale_multiple_images(runner):
    """Test the get-scale command with multiple images."""
    # Use all sample images that exist
    sample_images = [img for img in SAMPLE_IMAGES if img.exists()]
    if not sample_images:
        pytest.skip("No test images found")
    
    # Convert paths to strings for CLI command
    image_paths = [str(img) for img in sample_images]
    
    result = runner.invoke(app, [
        "scale", "get-scale", *image_paths, 
        "--verbose"  # Use verbose to get more output
    ])
    
    assert result.exit_code == 0
    # Check for processing message for each image
    for img in sample_images:
        assert f"Processing {img}" in result.stdout
    
    # Check for summary
    assert "Summary:" in result.stdout
    assert f"/{len(sample_images)}" in result.stdout


def test_scale_get_scale_debug_mode(runner):
    """Test the get-scale command with debug flag."""
    # Use a sample image
    sample_image = SAMPLE_IMAGES[0]
    # Skip if image doesn't exist
    if not sample_image.exists():
        pytest.skip(f"Test image not found: {sample_image}")
    
    result = runner.invoke(app, [
        "scale", "get-scale", str(sample_image),
        "--debug"
    ])
    
    assert result.exit_code == 0
    # Check for debug output
    assert "[DEBUG]" in result.stdout
    assert "Image loaded:" in result.stdout
    assert "Converting to grayscale if needed" in result.stdout
    assert "Attempting to detect scale bar" in result.stdout
    assert "[DEBUG] Scale bar detected successfully" in result.stdout
    
    # Verify the command completed successfully with proper summary output
    assert "Scale detected:" in result.stdout
    assert "μm/pixel" in result.stdout
    assert "Successfully detected: 1/1" in result.stdout


def test_v1_backward_compatibility(runner):
    """Test that legacy commands are available under v1 subcommand."""
    # Test help output for legacy commands
    legacy_commands = ["train", "predict", "analyze", "consolidate", "preprocess"]
    for cmd in legacy_commands:
        result = runner.invoke(app, ["v1", cmd, "--help"])
        assert result.exit_code == 0
        # Each command should have a proper help message
        assert result.stdout.strip() != ""


def test_missing_image_handling(runner):
    """Test that the CLI handles missing image files gracefully."""
    # Use a non-existent image path
    nonexistent_path = "nonexistent_image.tif"
    
    result = runner.invoke(app, [
        "scale", "get-scale", nonexistent_path
    ])
    
    assert result.exit_code == 0  # CLI should not crash
    assert "Error: Image file not found" in result.stdout
