"""UNet model implementation.

This module contains the UNet model implementation adapted to work with UNetConfig.
"""

from typing import TYPE_CHECKING

from tensorflow import Tensor
from tensorflow.keras import Model

from ..base import BaseModel
from .config import UNetConfig

if TYPE_CHECKING:
    from keras.src.layers import (
        BatchNormalization,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Input,
        LeakyReLU,
        MaxPooling2D,
        SpatialDropout2D,
    )
else:
    from tensorflow.keras.layers import (
        BatchNormalization,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Input,
        LeakyReLU,
        MaxPooling2D,
        SpatialDropout2D,
    )


def get_conv_block(x: Tensor, filters: int, dropout_rate: float, stage: str) -> Tensor:
    """Create a convolutional block.

    Args:
        x: Input tensor.
        filters: Number of filters.
        dropout_rate: Dropout rate.
        stage: Stage name for layer naming.

    Returns:
        Output tensor after convolution, activation, dropout and batch normalization.
    """
    kernel_size = 3

    x = Conv2D(filters, kernel_size, padding="same", name=f"{stage}_conv1")(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    if dropout_rate > 0.0:
        x = SpatialDropout2D(dropout_rate, name=f"{stage}_drop1")(x)
    x = BatchNormalization(name=f"{stage}_bnor1")(x)

    x = Conv2D(filters, kernel_size, padding="same", name=f"{stage}_conv2")(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    if dropout_rate > 0.0:
        x = SpatialDropout2D(dropout_rate, name=f"{stage}_drop2")(x)
    x = BatchNormalization(name=f"{stage}_bnor2")(x)

    return x


def get_upconv(x: Tensor, copy: Tensor, filters: int, stage: str) -> Tensor:
    """Create an up-convolutional block.

    Args:
        x: Input tensor.
        copy: Skip connection tensor.
        filters: Number of filters.
        stage: Stage name for layer naming.

    Returns:
        Output tensor after up-convolution and concatenation.
    """
    x = Conv2DTranspose(filters, 2, strides=2, padding="same", name=f"{stage}_upconv")(
        x
    )
    x = Concatenate(name=f"{stage}_concat")([x, copy])
    return x


class UNetModel(BaseModel):
    """UNet model implementation."""

    def __init__(self, config: UNetConfig):
        """Initialize the UNet model.

        Args:
            config: UNet configuration.
        """
        super().__init__(config)
        self.unet_config = config  # Type-specific reference for better IDE support

    def build(self) -> Model:
        """Build the UNet model architecture.

        Returns:
            Built Keras model.
        """
        # Extract configuration parameters
        input_shape = self.unet_config.input_shape
        dropout_rate = self.unet_config.dropout_rate
        depth = min(len(self.unet_config.filters), self.unet_config.depth)

        # Input layer
        inputs = Input(input_shape, name="input")
        x = inputs

        # Optional Gaussian noise for training robustness
        # x = GaussianNoise(0.1)(x)

        # Keep track of skip connections for the U-Net architecture
        copies = []

        # Downsampling path
        for i in range(depth):
            filters = self.unet_config.filters[i]
            stage = f"down{i + 1}"

            x = get_conv_block(x, filters, dropout_rate, stage)
            copies.append(x)

            if i < depth - 1:
                x = MaxPooling2D(2, name=f"{stage}_pool")(x)

        # Upsampling path
        for i in range(depth - 1):
            idx = depth - i - 2
            filters = self.unet_config.filters[idx]
            stage = f"up{idx + 1}"

            x = get_upconv(x, copies[idx], filters, stage)
            x = get_conv_block(x, filters, dropout_rate, stage)

        # Output layer
        outputs = Conv2D(
            1, 1, activation=self.unet_config.final_activation, name="output"
        )(x)

        # Create and return model
        model = Model(inputs, outputs, name="unet")
        return model
