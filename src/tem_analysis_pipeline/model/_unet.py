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


def get_conv_block(x: Tensor, filters: int, dropout_rate: float, stage: str) -> Tensor:
    kernel_size = 3

    x = Conv2D(filters, kernel_size, name=f"{stage}_conv1")(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    if dropout_rate > 0.0:
        x = SpatialDropout2D(dropout_rate, name=f"{stage}_drop1")(x)
    x = BatchNormalization(name=f"{stage}_bnor1")(x)

    x = Conv2D(filters, kernel_size, name=f"{stage}_conv2")(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    if dropout_rate > 0.0:
        x = SpatialDropout2D(dropout_rate, name=f"{stage}_drop2")(x)
    x = BatchNormalization(name=f"{stage}_bnor2")(x)

    return x


def get_upconv(x: Tensor, copy: Tensor, filters: int, stage: str) -> Tensor:
    x = Conv2DTranspose(filters, (2, 2), 2, name=f"{stage}_upsmp")(x)
    x = LeakyReLU(negative_slope=0.2)(x)

    ver_pad = (copy.shape[1] - x.shape[1]) // 2
    ver_res = (copy.shape[1] - x.shape[1]) % 2
    hor_pad = (copy.shape[2] - x.shape[2]) // 2
    hor_res = (copy.shape[2] - x.shape[2]) % 2

    crop = ((ver_pad, ver_pad + ver_res), (hor_pad, hor_pad + hor_res))
    paste = Cropping2D(crop, name=f"{stage}_paste")(copy)
    x = Concatenate(name=f"{stage}_input")([paste, x])
    return x


def make_unet(
    tile_shape: tuple[int, int],
    channels: int,
    layer_depth: int = 5,
    filters_root: int = 16,
) -> Model:
    input_shape = tile_shape + (channels,)

    filter_list = [filters_root * 2**i for i in range(layer_depth)]
    level_list = list(range(1, len(filter_list) + 1))
    copy_list = []

    for lvl, filters in zip(level_list, filter_list):
        if lvl == 1:
            x = input = Input(input_shape, name="contract1_input", dtype="float32")
            x = GaussianNoise(0.01)(x)
        else:
            x = MaxPooling2D((2, 2), name=f"contract{lvl}_input", strides=2)(x)

        x = get_conv_block(
            x,
            filters,
            dropout_rate=0.20,
            stage=f"contract{lvl}",
        )
        copy_list.append(x)

    for lvl, filters, copy in zip(level_list[::-1], filter_list[::-1], copy_list[::-1]):
        if lvl == max(level_list):
            continue

        x = get_upconv(x, copy, filters, stage=f"expand{lvl}")

        x = get_conv_block(
            x,
            filters,
            dropout_rate=0.10,
            stage=f"expand{lvl}",
        )

    x = Conv2D(1, 1, name="final_conv")(x)
    output = Activation("sigmoid", dtype="float32", name="output")(x)

    # Define the model
    model = Model(input, output)

    return model
