from typing import TYPE_CHECKING

from tensorflow import Tensor

if TYPE_CHECKING:
    from keras import Model
    from keras.src.layers import (
        Activation,
        BatchNormalization,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Cropping2D,
        GaussianNoise,
        Input,
        LeakyReLU,
        MaxPooling2D,
        SpatialDropout2D,
    )

else:
    from tensorflow.keras import Model
    from tensorflow.keras.layers import (
        Activation,
        BatchNormalization,
        Concatenate,
        Conv2D,
        Conv2DTranspose,
        Cropping2D,
        GaussianNoise,
        Input,
        LeakyReLU,
        MaxPooling2D,
        SpatialDropout2D,
    )


def _get_conv_block(x: Tensor, filters: int, dropout_rate: float, stage: str) -> Tensor:
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


def _get_upconv(x: Tensor, filters: int, stage: str) -> Tensor:
    x = Conv2DTranspose(filters, (2, 2), 2, name=f"{stage}_upsmp")(x)
    x = LeakyReLU(negative_slope=0.2)(x)
    return x


def _crop_and_concat(x: Tensor, copy: Tensor, stage: str) -> Tensor:
    ver_pad = (copy.shape[1] - x.shape[1]) // 2
    ver_res = (copy.shape[1] - x.shape[1]) % 2
    hor_pad = (copy.shape[2] - x.shape[2]) // 2
    hor_res = (copy.shape[2] - x.shape[2]) % 2

    crop = ((ver_pad, ver_pad + ver_res), (hor_pad, hor_pad + hor_res))
    paste = Cropping2D(crop, name=f"{stage}_paste")(copy)
    x = Concatenate(name=f"{stage}_concat")([paste, x])
    return x


def make_unet_plusplus(
    tile_shape: tuple[int, int],
    channels: int,
    layer_depth: int = 5,
    filters_root: int = 16,
    deep_supervision: bool = True,
) -> Model:
    input_shape = tile_shape + (channels,)

    # Initialize the nested feature map dictionary
    # X[i,j] represents the feature map at depth i and position j
    X = {}

    filter_list = [filters_root * 2**i for i in range(layer_depth)]

    # Input layer
    input = Input(input_shape, name="input", dtype="float32")
    X[0, 0] = GaussianNoise(0.01)(input)

    # Build the encoder path and nested skip connections
    for i in range(layer_depth):
        # Encoder path (leftmost column)
        if i == 0:
            X[i, 0] = _get_conv_block(
                X[i, 0], filter_list[i], dropout_rate=0.20, stage=f"encode_{i}_{0}"
            )
        else:
            down = MaxPooling2D((2, 2), strides=2, name=f"pool_{i}_{0}")(X[i - 1, 0])
            X[i, 0] = _get_conv_block(
                down, filter_list[i], dropout_rate=0.20, stage=f"encode_{i}_{0}"
            )

    tensor: Tensor
    concat_list: list[Tensor]

    # Build nested skip pathways
    for i in range(layer_depth - 1):
        for j in range(1, layer_depth - i):
            # Collect all incoming connections
            up = _get_upconv(X[i + 1, j - 1], filter_list[i], stage=f"up_{i}_{j}")

            # Concatenate with all previous horizontal connections
            concat_list = [X[i, k] for k in range(j)]
            concat_list.append(up)

            # Handle cropping for each connection
            # Find minimum size among all tensors to concatenate
            min_height = min(t.shape[1] for t in concat_list)
            min_width = min(t.shape[2] for t in concat_list)

            # Crop all tensors to minimum size
            cropped_list = []
            for idx, tensor in enumerate(concat_list):
                if tensor.shape[1] > min_height or tensor.shape[2] > min_width:
                    ver_pad = (tensor.shape[1] - min_height) // 2
                    ver_res = (tensor.shape[1] - min_height) % 2
                    hor_pad = (tensor.shape[2] - min_width) // 2
                    hor_res = (tensor.shape[2] - min_width) % 2

                    crop = ((ver_pad, ver_pad + ver_res), (hor_pad, hor_pad + hor_res))
                    tensor = Cropping2D(crop, name=f"crop_{i}_{j}_{idx}")(tensor)
                cropped_list.append(tensor)

            # Concatenate all connections
            if len(cropped_list) > 1:
                concat = Concatenate(name=f"concat_{i}_{j}")(cropped_list)
            else:
                concat = cropped_list[0]

            # Apply convolution block
            X[i, j] = _get_conv_block(
                concat, filter_list[i], dropout_rate=0.10, stage=f"dense_{i}_{j}"
            )

    # Output layers
    if deep_supervision:
        # Create outputs at different scales
        outputs = []
        for j in range(1, layer_depth):
            out = Conv2D(1, 1, name=f"final_conv_{j}")(X[0, j])
            out = Activation("sigmoid", dtype="float32", name=f"output_{j}")(out)
            outputs.append(out)

        # Define the model with multiple outputs
        model = Model(input, outputs)
    else:
        # Single output from the final nested block
        output = Conv2D(1, 1, name="final_conv")(X[0, layer_depth - 1])
        output = Activation("sigmoid", dtype="float32", name="output")(output)

        # Define the model
        model = Model(input, output)

    return model
