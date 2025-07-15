"""Scale Estimation System for TEM Image Analysis.

This module provides abstractions for estimating the scale of TEM images,
allowing for different scale estimation strategies and fallback mechanisms.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image


class ScaleEstimationError(Exception):
    """Base exception for scale estimation errors."""

    pass


class NoScaleFoundError(ScaleEstimationError):
    """Raised when no scale information can be found in the image."""

    pass


class ScaleDetectionFailureError(ScaleEstimationError):
    """Raised when scale detection algorithm fails."""

    pass


class InvalidInputError(ScaleEstimationError):
    """Raised when input is not a valid image or cannot be processed."""

    pass


class ScaleEstimator(ABC):
    """Abstract base class for image scale estimation.

    This class defines the interface for scale estimators that can
    determine the scale (in micrometers per pixel) from an input image.
    Implementations should handle different input types and provide
    robust error handling.
    """

    @abstractmethod
    def estimate_scale(
        self, image_input: Union[Path, Image.Image, np.ndarray]
    ) -> float:
        """Estimate the scale of an image in micrometers per pixel.

        Args:
            image_input: Image to estimate scale from. Can be a path to an image file,
                         a PIL Image object, or a numpy array.

        Returns:
            float: Estimated scale in micrometers per pixel.

        Raises:
            NoScaleFoundError: If no scale information can be found in the image.
            ScaleDetectionFailureError: If scale detection algorithm fails.
            InvalidInputError: If input is not a valid image or cannot be processed.
        """
        pass

    @abstractmethod
    def can_estimate(self, image_input: Union[Path, Image.Image, np.ndarray]) -> bool:
        """Check if this estimator can handle the given image.

        This method should quickly determine if the estimator is likely to
        successfully extract scale information from the image, without
        doing the full estimation calculation.

        Args:
            image_input: Image to check. Can be a path to an image file,
                        a PIL Image object, or a numpy array.

        Returns:
            bool: True if this estimator can likely handle the image, False otherwise.
        """
        pass

    def _convert_to_pil(
        self, image_input: Union[Path, Image.Image, np.ndarray]
    ) -> Image.Image:
        """Convert various input types to PIL Image in "L" mode (8-bit grayscale).

        Args:
            image_input: Input image in various formats.

        Returns:
            Image.Image: PIL Image object in "L" mode.

        Raises:
            InvalidInputError: If input cannot be converted to PIL Image.
        """
        try:
            # Convert to PIL Image based on input type
            if isinstance(image_input, Path):
                img = Image.open(image_input)
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                img = image_input
            else:
                raise InvalidInputError(f"Unsupported input type: {type(image_input)}")

            # Ensure image is in "L" mode (8-bit grayscale)
            if img.mode != "L":
                try:
                    x = np.array(img)
                    x = ((x / np.iinfo(x.dtype).max) * 255).round().astype(np.uint8)
                    img = Image.fromarray(x)

                except Exception as e:
                    raise InvalidInputError(f"Failed to convert image to 'L' mode: {e}")

            return img
        except InvalidInputError:
            raise
        except Exception as e:
            raise InvalidInputError(f"Failed to convert input to PIL Image: {e}")


class ScaleBarReader(ScaleEstimator):
    """Scale estimator that reads scale bars in TEM images.

    This class implements the ScaleEstimator interface to detect and read
    scale bars commonly found in TEM images. It uses the existing calibration
    code from the TEM analysis pipeline.
    """

    def estimate_scale(
        self, image_input: Union[Path, Image.Image, np.ndarray]
    ) -> float:
        """Estimate scale from scale bar in image.

        Args:
            image_input: Image to estimate scale from.

        Returns:
            float: Estimated scale in micrometers per pixel.

        Raises:
            NoScaleFoundError: If no scale bar can be found in the image.
            ScaleDetectionFailureError: If scale bar detection fails.
        """
        from .calibration import (
            get_calibration,
            NoScaleError,
            NoScaleNumberError,
            NoScaleUnitError,
        )

        img = self._convert_to_pil(image_input)

        try:
            return get_calibration(img)
        except NoScaleError as e:
            raise NoScaleFoundError(f"Could not find scale bar: {e}")
        except (NoScaleNumberError, NoScaleUnitError) as e:
            raise ScaleDetectionFailureError(f"Failed to read scale: {e}")
        except Exception as e:
            raise ScaleDetectionFailureError(
                f"Unexpected error in scale detection: {e}"
            )

    def can_estimate(self, image_input: Union[Path, Image.Image, np.ndarray]) -> bool:
        """Check if image likely contains a scale bar that can be read.

        This implementation attempts to perform the full scale estimation
        and returns True if it succeeds, False otherwise.

        Args:
            image_input: Image to check.

        Returns:
            bool: True if scale bar can be detected, False otherwise.
        """
        try:
            self.estimate_scale(image_input)
            return True
        except ScaleEstimationError:
            return False


class ScaleManager:
    """Manages scale estimation with fallback strategies.

    This class orchestrates multiple scale estimators, trying them in sequence
    until one succeeds or all fail. It also supports fallback to a default scale.
    """

    def __init__(self, estimators: Optional[List[ScaleEstimator]] = None):
        """Initialize with list of estimators.

        Args:
            estimators: List of scale estimators to use, in order of preference.
                       If None, uses ScaleBarReader as the default estimator.
        """
        self.estimators = estimators or [ScaleBarReader()]

    def get_scale(
        self,
        image_input: Union[Path, Image.Image, np.ndarray],
        fallback: Optional[float] = None,
    ) -> float:
        """Try estimators in order, return first successful result.

        Args:
            image_input: Image to estimate scale from.
            fallback: Optional fallback scale to use if all estimators fail.

        Returns:
            float: Estimated scale in micrometers per pixel.

        Raises:
            ScaleEstimationError: If all estimators fail and no fallback is provided.
        """
        errors = []

        # First try estimators that report they can handle the image
        for estimator in self.estimators:
            if estimator.can_estimate(image_input):
                try:
                    return estimator.estimate_scale(image_input)
                except ScaleEstimationError as e:
                    errors.append(str(e))

        # If no estimator worked and we have a fallback, use it
        if fallback is not None:
            return fallback

        # No estimator worked and no fallback, raise error with details
        error_details = "\n".join(errors) if errors else "No suitable estimator found"
        raise ScaleEstimationError(f"Could not estimate scale: {error_details}")

    def register_estimator(self, estimator: ScaleEstimator, priority: int = -1) -> None:
        """Register a new estimator.

        Args:
            estimator: Scale estimator to register.
            priority: Position to insert estimator at. Negative values append to the end.
        """
        if priority < 0 or priority >= len(self.estimators):
            self.estimators.append(estimator)
        else:
            self.estimators.insert(priority, estimator)
