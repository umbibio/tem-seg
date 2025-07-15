"""Tests for the Scale Estimation System."""

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from PIL import Image

from tem_analysis_pipeline.scale_estimation import (
    NoScaleFoundError,
    ScaleBarReader,
    ScaleEstimationError,
    ScaleManager,
    InvalidInputError,
)


class TestScaleBarReader(unittest.TestCase):
    """Tests for the ScaleBarReader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.reader = ScaleBarReader()

        # Create a mock image with a scale bar
        self.mock_image = Image.new("L", (800, 600), color=255)

    @mock.patch("tem_analysis_pipeline.scale_estimation.ScaleBarReader._convert_to_pil")
    @mock.patch("tem_analysis_pipeline.calibration.get_calibration")
    def test_estimate_scale_success(self, mock_get_calibration, mock_convert_to_pil):
        """Test successful scale estimation."""
        # Setup mocks
        mock_img = mock.MagicMock()
        mock_convert_to_pil.return_value = mock_img
        mock_get_calibration.return_value = 0.05  # 0.05 μm per pixel

        # Call method
        result = self.reader.estimate_scale("mock_path")

        # Check results
        mock_convert_to_pil.assert_called_once_with("mock_path")
        mock_get_calibration.assert_called_once_with(mock_img)
        assert result == 0.05

    @mock.patch("tem_analysis_pipeline.scale_estimation.ScaleBarReader._convert_to_pil")
    @mock.patch("tem_analysis_pipeline.calibration.get_calibration")
    def test_estimate_scale_no_scale_found(
        self, mock_get_calibration, mock_convert_to_pil
    ):
        """Test no scale found error."""
        # Setup mocks
        from tem_analysis_pipeline.calibration import NoScaleError

        mock_img = mock.MagicMock()
        mock_convert_to_pil.return_value = mock_img
        mock_get_calibration.side_effect = NoScaleError("Scale bar not found")

        # Call method and check exception
        with pytest.raises(NoScaleFoundError):
            self.reader.estimate_scale("mock_path")

    def test_convert_to_pil_path(self):
        """Test conversion from Path to PIL Image."""
        # This test would normally use a real image file
        # For now, we'll mock the Image.open method
        with mock.patch("PIL.Image.open") as mock_open:
            mock_img = mock.MagicMock()
            # Configure mock image to have mode property and convert method
            mock_img.mode = "L"  # Already in grayscale mode
            mock_img.convert.return_value = mock_img  # Return self from convert
            mock_open.return_value = mock_img

            path = Path("/path/to/image.png")
            result = self.reader._convert_to_pil(path)

            mock_open.assert_called_once_with(path)
            # Since mode is already "L", convert shouldn't be called
            mock_img.convert.assert_not_called()
            assert result == mock_img

    def test_convert_to_pil_array(self):
        """Test conversion from numpy array to PIL Image."""
        # Create a simple numpy array
        array = np.zeros((10, 10), dtype=np.uint8)

        result = self.reader._convert_to_pil(array)

        assert isinstance(result, Image.Image)
        assert result.size == (10, 10)
        assert result.mode == "L"

    def test_convert_to_pil_image(self):
        """Test conversion from PIL Image (no conversion needed)."""
        img = Image.new("L", (10, 10), color=0)

        result = self.reader._convert_to_pil(img)

        assert result is img  # Should be the same object

    def test_convert_to_pil_invalid(self):
        """Test conversion from invalid type raises error."""
        with pytest.raises(InvalidInputError):
            self.reader._convert_to_pil("not an image")

    def test_can_estimate(self):
        """Test can_estimate method."""
        with mock.patch.object(self.reader, "estimate_scale") as mock_estimate:
            # Test when estimation succeeds
            mock_estimate.return_value = 0.05
            assert self.reader.can_estimate("mock_path") is True

            # Test when estimation fails
            mock_estimate.side_effect = NoScaleFoundError("No scale found")
            assert self.reader.can_estimate("mock_path") is False


class TestScaleManager(unittest.TestCase):
    """Tests for the ScaleManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.estimator1 = mock.MagicMock()
        self.estimator2 = mock.MagicMock()
        self.manager = ScaleManager([self.estimator1, self.estimator2])

    def test_get_scale_first_estimator_success(self):
        """Test get_scale with first estimator succeeding."""
        self.estimator1.can_estimate.return_value = True
        self.estimator1.estimate_scale.return_value = 0.05

        result = self.manager.get_scale("mock_path")

        assert result == 0.05
        self.estimator1.can_estimate.assert_called_once_with("mock_path")
        self.estimator1.estimate_scale.assert_called_once_with("mock_path")
        self.estimator2.can_estimate.assert_not_called()

    def test_get_scale_second_estimator_success(self):
        """Test get_scale with second estimator succeeding after first fails."""
        self.estimator1.can_estimate.return_value = False
        self.estimator2.can_estimate.return_value = True
        self.estimator2.estimate_scale.return_value = 0.1

        result = self.manager.get_scale("mock_path")

        assert result == 0.1
        self.estimator1.can_estimate.assert_called_once_with("mock_path")
        self.estimator1.estimate_scale.assert_not_called()
        self.estimator2.can_estimate.assert_called_once_with("mock_path")
        self.estimator2.estimate_scale.assert_called_once_with("mock_path")

    def test_get_scale_estimator_says_can_but_fails(self):
        """Test get_scale when estimator says it can estimate but then fails."""
        self.estimator1.can_estimate.return_value = True
        self.estimator1.estimate_scale.side_effect = ScaleEstimationError("Failed")
        self.estimator2.can_estimate.return_value = True
        self.estimator2.estimate_scale.return_value = 0.1

        result = self.manager.get_scale("mock_path")

        assert result == 0.1
        self.estimator1.can_estimate.assert_called_once_with("mock_path")
        self.estimator1.estimate_scale.assert_called_once_with("mock_path")
        self.estimator2.can_estimate.assert_called_once_with("mock_path")
        self.estimator2.estimate_scale.assert_called_once_with("mock_path")

    def test_get_scale_all_fail_with_fallback(self):
        """Test get_scale when all estimators fail but fallback is provided."""
        self.estimator1.can_estimate.return_value = False
        self.estimator2.can_estimate.return_value = False

        result = self.manager.get_scale("mock_path", fallback=0.15)

        assert result == 0.15
        self.estimator1.can_estimate.assert_called_once_with("mock_path")
        self.estimator2.can_estimate.assert_called_once_with("mock_path")

    def test_get_scale_all_fail_no_fallback(self):
        """Test get_scale when all estimators fail and no fallback is provided."""
        self.estimator1.can_estimate.return_value = False
        self.estimator2.can_estimate.return_value = False

        with pytest.raises(ScaleEstimationError):
            self.manager.get_scale("mock_path")

    def test_register_estimator(self):
        """Test registering a new estimator."""
        new_estimator = mock.MagicMock()

        # Test appending
        self.manager.register_estimator(new_estimator)
        assert len(self.manager.estimators) == 3
        assert self.manager.estimators[2] == new_estimator

        # Test inserting at specific position
        another_estimator = mock.MagicMock()
        self.manager.register_estimator(another_estimator, priority=1)
        assert len(self.manager.estimators) == 4
        assert self.manager.estimators[1] == another_estimator


if __name__ == "__main__":
    unittest.main()
