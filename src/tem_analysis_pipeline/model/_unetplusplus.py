from typing import TYPE_CHECKING
from tensorflow import Tensor
from tensorflow.keras import Model

if TYPE_CHECKING:
    from keras.src.layers import (
        Activation,
        BatchNormalization,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Input,
        LeakyReLU,
        MaxPooling2D,
        SpatialDropout2D,
        UpSampling2D,
    )
else:
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Input,
        LeakyReLU,
        MaxPooling2D,
        SpatialDropout2D,
        UpSampling2D,
    )


def conv_block(x: Tensor, filters: int, dropout_rate: float, stage: str) -> Tensor:
    """Standard convolution block with two Conv2D layers."""
    x = Conv2D(filters, 3, padding='same', name=f"{stage}_conv1")(x)
    x = BatchNormalization(name=f"{stage}_bn1")(x)
    x = LeakyReLU(negative_slope=0.2, name=f"{stage}_relu1")(x)
    
    if dropout_rate > 0.0:
        x = SpatialDropout2D(dropout_rate, name=f"{stage}_drop1")(x)
    
    x = Conv2D(filters, 3, padding='same', name=f"{stage}_conv2")(x)
    x = BatchNormalization(name=f"{stage}_bn2")(x)
    x = LeakyReLU(negative_slope=0.2, name=f"{stage}_relu2")(x)
    
    if dropout_rate > 0.0:
        x = SpatialDropout2D(dropout_rate, name=f"{stage}_drop2")(x)
    
    return x


def make_unet_plus_plus(
    input_shape: tuple[int, int, int],
    num_classes: int,
    dropout_rate: float = 0.0,
    deep_supervision: bool = False,
) -> Model:
    """
    UNet++ (UNet Plus Plus) implementation.
    
    Args:
        input_shape: Input tensor shape (height, width, channels)
        num_classes: Number of output classes
        dropout_rate: Dropout rate for regularization
        deep_supervision: Whether to use deep supervision with multiple outputs
    
    Returns:
        Keras Model instance
    """
    inputs = Input(shape=input_shape, name="input")
    
    # Filter sizes for each level
    filters = [32, 64, 128, 256, 512]
    
    # Storage for skip connections - nested structure
    # x[i][j] represents the node at level i, position j
    x = {}
    
    # Encoder path (leftmost column)
    x[0] = {}
    x[0][0] = conv_block(inputs, filters[0], dropout_rate, "x00")
    
    x[1] = {}
    x[1][0] = MaxPooling2D((2, 2), name="pool1")(x[0][0])
    x[1][0] = conv_block(x[1][0], filters[1], dropout_rate, "x10")
    
    x[2] = {}
    x[2][0] = MaxPooling2D((2, 2), name="pool2")(x[1][0])
    x[2][0] = conv_block(x[2][0], filters[2], dropout_rate, "x20")
    
    x[3] = {}
    x[3][0] = MaxPooling2D((2, 2), name="pool3")(x[2][0])
    x[3][0] = conv_block(x[3][0], filters[3], dropout_rate, "x30")
    
    x[4] = {}
    x[4][0] = MaxPooling2D((2, 2), name="pool4")(x[3][0])
    x[4][0] = conv_block(x[4][0], filters[4], dropout_rate, "x40")
    
    # Dense skip connections - built in correct order (bottom-up, left-to-right)
    
    # Level 3 (bottom level first)
    x[3][1] = Conv2DTranspose(filters[3], (2, 2), strides=2, padding='same', name="up_x31")(x[4][0])
    x[3][1] = Concatenate(name="concat_x31")([x[3][0], x[3][1]])
    x[3][1] = conv_block(x[3][1], filters[3], dropout_rate, "x31")
    
    # Level 2
    x[2][1] = Conv2DTranspose(filters[2], (2, 2), strides=2, padding='same', name="up_x21")(x[3][0])
    x[2][1] = Concatenate(name="concat_x21")([x[2][0], x[2][1]])
    x[2][1] = conv_block(x[2][1], filters[2], dropout_rate, "x21")
    
    x[2][2] = Conv2DTranspose(filters[2], (2, 2), strides=2, padding='same', name="up_x22")(x[3][1])
    x[2][2] = Concatenate(name="concat_x22")([x[2][0], x[2][1], x[2][2]])
    x[2][2] = conv_block(x[2][2], filters[2], dropout_rate, "x22")
    
    # Level 1
    x[1][1] = Conv2DTranspose(filters[1], (2, 2), strides=2, padding='same', name="up_x11")(x[2][0])
    x[1][1] = Concatenate(name="concat_x11")([x[1][0], x[1][1]])
    x[1][1] = conv_block(x[1][1], filters[1], dropout_rate, "x11")
    
    x[1][2] = Conv2DTranspose(filters[1], (2, 2), strides=2, padding='same', name="up_x12")(x[2][1])
    x[1][2] = Concatenate(name="concat_x12")([x[1][0], x[1][1], x[1][2]])
    x[1][2] = conv_block(x[1][2], filters[1], dropout_rate, "x12")
    
    x[1][3] = Conv2DTranspose(filters[1], (2, 2), strides=2, padding='same', name="up_x13")(x[2][2])
    x[1][3] = Concatenate(name="concat_x13")([x[1][0], x[1][1], x[1][2], x[1][3]])
    x[1][3] = conv_block(x[1][3], filters[1], dropout_rate, "x13")
    
    # Level 0 (top row)
    x[0][1] = Conv2DTranspose(filters[0], (2, 2), strides=2, padding='same', name="up_x01")(x[1][0])
    x[0][1] = Concatenate(name="concat_x01")([x[0][0], x[0][1]])
    x[0][1] = conv_block(x[0][1], filters[0], dropout_rate, "x01")
    
    x[0][2] = Conv2DTranspose(filters[0], (2, 2), strides=2, padding='same', name="up_x02")(x[1][1])
    x[0][2] = Concatenate(name="concat_x02")([x[0][0], x[0][1], x[0][2]])
    x[0][2] = conv_block(x[0][2], filters[0], dropout_rate, "x02")
    
    x[0][3] = Conv2DTranspose(filters[0], (2, 2), strides=2, padding='same', name="up_x03")(x[1][2])
    x[0][3] = Concatenate(name="concat_x03")([x[0][0], x[0][1], x[0][2], x[0][3]])
    x[0][3] = conv_block(x[0][3], filters[0], dropout_rate, "x03")
    
    x[0][4] = Conv2DTranspose(filters[0], (2, 2), strides=2, padding='same', name="up_x04")(x[1][3])
    x[0][4] = Concatenate(name="concat_x04")([x[0][0], x[0][1], x[0][2], x[0][3], x[0][4]])
    x[0][4] = conv_block(x[0][4], filters[0], dropout_rate, "x04")
    
    # Output layer(s)
    if deep_supervision:
        # Multiple outputs for deep supervision
        output1 = Conv2D(num_classes, 1, activation='softmax', name="output1")(x[0][1])
        output2 = Conv2D(num_classes, 1, activation='softmax', name="output2")(x[0][2])
        output3 = Conv2D(num_classes, 1, activation='softmax', name="output3")(x[0][3])
        output4 = Conv2D(num_classes, 1, activation='softmax', name="output4")(x[0][4])
        
        model = Model(inputs=inputs, outputs=[output1, output2, output3, output4])
    else:
        # Single output (usually the final one with richest features)
        output = Conv2D(num_classes, 1, activation='softmax', name="output")(x[0][4])
        model = Model(inputs=inputs, outputs=output)
    
    return model


# Example usage:
if __name__ == "__main__":
    # For binary segmentation
    model = make_unet_plus_plus(
        input_shape=(256, 256, 3),
        num_classes=2,
        dropout_rate=0.1,
        deep_supervision=False
    )
    
    model.summary()
    
    # With deep supervision (multiple outputs)
    model_deep = make_unet_plus_plus(
        input_shape=(256, 256, 3),
        num_classes=2,
        dropout_rate=0.1,
        deep_supervision=True
    )
    
    print("\nWith deep supervision:")
    model_deep.summary()