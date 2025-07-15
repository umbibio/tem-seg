"""UNet model configuration.

This module defines the configuration parameters specific to the UNet architecture.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from ..base import BaseModelConfig


@dataclass(frozen=True)
class UNetConfig(BaseModelConfig):
    """Configuration for UNet architecture.

    Attributes:
        input_shape: Input shape as (height, width, channels).
        filters: List of filter sizes for each level of the UNet.
        kernel_size: Kernel size for convolutional layers.
        pool_size: Pool size for max pooling layers.
        depth: Depth of the UNet (number of downsampling/upsampling steps).
        use_attention: Whether to use attention gates in the skip connections.
    """

    input_shape: Tuple[int, int, int] = (256, 256, 1)
    filters: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    kernel_size: Tuple[int, int] = (3, 3)
    pool_size: Tuple[int, int] = (2, 2)
    depth: int = 4
    use_attention: bool = False

    @property
    def architecture(self) -> str:
        """Return the architecture name."""
        return "unet"
