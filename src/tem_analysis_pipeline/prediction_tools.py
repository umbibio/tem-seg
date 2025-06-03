import os
from pathlib import Path
import sys
import itertools
from glob import glob

import cv2
import numpy as np
import scipy as sp

from PIL import Image

from .calibration import fix_image_scale


CONFIG = {}


def select_model_version(
    model_version: str,
    *,
    models_folder: str | Path = None,
    use_ensemble: bool = False,
):
    this = Path(__file__)
    if models_folder is not None:
        models_folder = Path(models_folder)
    else:
        models_folder = this.parent.parent.parent / "models"

    if use_ensemble:
        model_root = models_folder / "cross_validation" / model_version
    else:
        model_root = models_folder / "single_fold" / model_version

    assert model_root.exists(), model_root
    assert model_root.is_dir(), model_root

    CONFIG["model_root"] = model_root.as_posix()
    sys.path.insert(1, model_root.as_posix())


def build_ensemble(models):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import backend

    from .model.custom_objects import MyWeightedBinaryCrossEntropy, MyMeanDSC, MyMeanIoU

    m = models[0]
    if hasattr(m, "input"):
        input_shape = m.input.shape[1:]
    elif hasattr(m, "input_layer"):
        input_shape = m.input_layer.shape[1:]
    else:
        raise ValueError("Model has no valid input")

    inputs = tf.keras.Input(shape=input_shape)
    for model in models:
        model.trainable = False

    ys = []
    for i, model in enumerate(models):
        inp = tf.keras.Input(shape=input_shape)
        out = model(inp, training=False)
        wrap = tf.keras.Model(inputs=inp, outputs=out, name=f"model_fold_{i + 1}")

        y = wrap(inputs)
        ys.append(y)

    x = layers.Average()(ys)
    outputs = layers.Lambda(backend.round)(x)

    ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    loss_pos_weight = 2
    loss = MyWeightedBinaryCrossEntropy(pos_weight=loss_pos_weight)
    mean_iou_metric = MyMeanIoU(name="mean_iou")
    mean_dsc_metric = MyMeanDSC(name="dice_coefficient")
    metrics = [mean_iou_metric, mean_dsc_metric, "Recall"]

    ensemble_model.compile(
        loss=loss,
        optimizer="adam",
        metrics=metrics,
    )
    return ensemble_model


def load_nontrainable_model(p, round_output=False):
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras import backend

    from .model.custom_objects import (
        custom_objects,
        MyWeightedBinaryCrossEntropy,
        MyMeanDSC,
        MyMeanIoU,
    )

    model = tf.keras.models.load_model(p, custom_objects=custom_objects)
    if round_output:
        output = layers.Lambda(backend.round)(model.output)
        model = tf.keras.Model(model.input, output)

        loss_pos_weight = 2
        loss = MyWeightedBinaryCrossEntropy(pos_weight=loss_pos_weight)
        mean_iou_metric = MyMeanIoU(name="mean_iou")
        mean_dsc_metric = MyMeanDSC(name="dice_coefficient")
        metrics = [mean_iou_metric, mean_dsc_metric, "Recall"]

        model.compile(
            loss=loss,
            optimizer="adam",
            metrics=metrics,
        )

    model.trainable = False
    return model


def get_list_of_models(organelle, ckpt="last", round_output=False):
    model_root = CONFIG["model_root"]
    return [
        load_nontrainable_model(p, round_output=round_output)
        for p in sorted(
            glob(os.path.join(model_root, f"{organelle}/kf??/ckpt/{ckpt}.h5"))
        )
    ]


def get_organelle_ensemble_model(organelle, ckpt="last"):
    models = get_list_of_models(organelle, ckpt=ckpt, round_output=False)
    return build_ensemble(models)


def get_ensembles(organelles, ckpt="last"):
    ensembles = {}
    for organelle in organelles:
        ensembles[organelle] = get_organelle_ensemble_model(organelle, ckpt=ckpt)
    return ensembles


def load_image(img_filepath: str) -> Image.Image:
    img = Image.open(img_filepath)
    if img.mode != "L":
        x = np.array(img)
        x = ((x / np.iinfo(x.dtype).max) * 255).round().astype(np.uint8)
        img = Image.fromarray(x)
    img.filename = img_filepath
    return img


def composition(img: Image.Image, layers: dict = {}) -> Image.Image:
    assert img.mode == "L"
    img = np.array(img, dtype=int)

    for key, msk in layers.items():
        assert msk.mode == "L"
        layers[key] = np.array(msk, dtype=int)

    r, g, b = img.copy(), img.copy(), img.copy()
    for key, array in layers.items():
        if key == "red" or key == "r":
            r = np.minimum(255, r + array)
        elif key == "green" or key == "g":
            g = np.minimum(255, g + array)
        elif key == "blue" or key == "b":
            b = np.minimum(255, b + array)
        elif key == "cyan" or key == "c":
            g = np.minimum(255, g + array)
            b = np.minimum(255, b + array)
        elif key == "magenta" or key == "m":
            r = np.minimum(255, r + array)
            b = np.minimum(255, b + array)
        elif key == "yellow" or key == "y":
            r = np.minimum(255, r + array)
            g = np.minimum(255, g + array)

    r = Image.fromarray(r.astype(np.uint8))
    g = Image.fromarray(g.astype(np.uint8))
    b = Image.fromarray(b.astype(np.uint8))

    return Image.merge("RGB", (r, g, b))


def image_prediction(
    img: Image.Image, model, trg_scale: float, batch_size=32, verbose=2
) -> Image.Image:
    """Predicts image using model.

    Parameters
    ----------
    img : Image.Image
        Image to use for segmentation prediction.
    model : tf.keras.Model
        Model to use for prediction.
    trg_scale : float
        Scale of the image in during prediction (units in um/px).
    batch_size : int, optional
        Batch size to use for prediction, by default 32
    verbose : int, optional
        Verbosity level, by default 2

    Returns
    -------
    Image.Image
        Predicted segmentation mask for input image.
    """
    import tensorflow as tf

    assert img.mode == "L"

    img_original_size = img.size
    img = fix_image_scale(img, trg_scale=trg_scale)
    img_arr = np.array(img).astype(np.float32) / 255

    try:
        model_input = model.input
    except AttributeError:
        model_input = model.input_layer
    # model_input = getattr(model, 'input', False) or getattr(model, 'input_layer')
    input_shape = tuple(model_input.shape[1:3])
    x = np.zeros((1,) + tuple(model_input.shape[1:]))
    output_shape = tuple(model(x).shape[1:3])

    input_size_y, input_size_x = input_shape
    output_size_y, output_size_x = output_shape

    padding_y = input_size_y // 2
    padding_x = input_size_x // 2

    io_offset_y = input_size_y - output_size_y
    io_offset_x = input_size_x - output_size_x

    io_step_offset_y = io_offset_y // 2
    io_step_offset_x = io_offset_x // 2

    img_arr_padded = np.pad(
        img_arr, ((padding_y, padding_y), (padding_x, padding_x)), "constant"
    )

    step_y = input_size_y // 4
    step_x = input_size_x // 4

    img_arr_tiles = np.lib.stride_tricks.sliding_window_view(
        img_arr_padded, input_shape
    )[::step_y, ::step_x]
    ny, nx = img_arr_tiles.shape[:2]
    img_arr_tiles = img_arr_tiles.reshape((-1,) + input_shape + (1,))
    img_arr_tiles_dataset = tf.data.Dataset.from_tensor_slices(img_arr_tiles).batch(
        batch_size
    )
    prd_arr_tiles = model.predict(
        img_arr_tiles_dataset, batch_size=batch_size, verbose=verbose
    )[..., 0]

    def stich_tiles(arr_tiles, offset_y=0, offset_x=0):
        tile_shape = arr_tiles.shape[1:]
        tile_size_y, tile_size_x = tile_shape
        canvas = np.zeros_like(img_arr_padded)
        counts = np.zeros_like(img_arr_padded)
        for idx, (j, i) in enumerate(itertools.product(np.arange(ny), np.arange(nx))):
            y, x = j * step_y + offset_y, i * step_x + offset_x
            canvas[y : y + tile_size_y, x : x + tile_size_x] += arr_tiles[idx]
            counts[y : y + tile_size_y, x : x + tile_size_x] += 1
        canvas = canvas[padding_y:-padding_y, padding_x:-padding_x]
        counts = counts[padding_y:-padding_y, padding_x:-padding_x]
        return canvas / counts

    output_arr = stich_tiles(
        prd_arr_tiles, offset_y=io_step_offset_y, offset_x=io_step_offset_x
    )
    output_arr = (output_arr * 255).round().astype(np.uint8)

    output = Image.fromarray(output_arr)

    return output.resize(img_original_size, resample=Image.Resampling.LANCZOS)


def threshold_prediction(img: Image.Image, threshold: float):
    """Threshold prediction to binary image.

    Parameters
    ----------
    img : Image.Image
        Image to threshold.
    threshold : float
        Threshold value in range [0, 1].

    Returns
    -------
    Image.Image
        Thresholded image.
    """

    assert img.mode == "L"

    threshold *= 255

    arr = np.array(img)
    arr[arr < threshold] = 0
    arr[arr >= threshold] = 255

    return Image.fromarray(arr)


def remove_small_predictions(
    prd: Image.Image, orig_scale: float, min_area_um2: int = 0.018
):
    """Removes small predictions from prediction image.

    Parameters
    ----------
    prd : Image.Image
        Prediction image. This should be a binary image. i.e. with pixel values 0 or 255.
    orig_scale : float
        Scale of the input image (units in um/px).
    min_area_um2 : int, optional
        Minimum area of prediction to keep, by default 0.05

    Returns
    -------
    Image.Image
        Prediction image with small predictions removed.
    """
    assert prd.mode == "L"
    min_area_px2 = min_area_um2 / orig_scale**2

    arr = np.array(prd)
    contours = cv2.findContours(arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area_px2:
            cv2.drawContours(arr, [cnt], 0, 0, -1)

    return Image.fromarray(arr)


def _gkern(sig=0.8):
    """creates gaussian kernel with side length `l` and a sigma of `sig`"""
    # https://stackoverflow.com/questions/29731726
    l = round(sig * 5)
    l += l % 2 + 1
    ax = np.linspace(-(l - 1) / 2.0, (l - 1) / 2.0, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def convolve_prediction(
    img: Image.Image,
    orig_scale: float,
    pred_scale: float,
    sigma_um: float = 0.005,
    n: int = 24,
):
    orig_size = img.size

    scale_ratio = pred_scale / orig_scale
    pred_width = round(img.width / scale_ratio)
    pred_height = round(img.height / scale_ratio)
    pred_size = (pred_width, pred_height)
    img = img.resize(pred_size, resample=Image.Resampling.NEAREST)
    cnv = np.array(img, dtype=float)

    sigma_px = sigma_um / pred_scale

    # https://stackoverflow.com/questions/33548639
    weights = _gkern(sigma_px)
    for _ in range(n):
        cnv = sp.ndimage.convolve(cnv, weights, mode="reflect")
    img = Image.fromarray(cnv.round().astype(np.uint8))

    return img.resize(orig_size, resample=Image.Resampling.LANCZOS)


def change_scale(
    img: Image.Image,
    orig_scale: float,
    pred_scale: float,
    resample=Image.Resampling.NEAREST,
):
    scale_ratio = pred_scale / orig_scale
    pred_width = round(img.width / scale_ratio)
    pred_height = round(img.height / scale_ratio)
    pred_size = (pred_width, pred_height)

    return img.resize(pred_size, resample=resample)


def test_convolution():
    import tensorflow as tf
    import numpy as np
    import scipy as sp

    arr = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    lbl = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    arr = np.pad(arr, 100, "constant", constant_values=0)
    lbl = np.pad(lbl, 100, "constant", constant_values=0)
    shape = arr.shape

    x_in = tf.constant(arr.reshape((1,) + shape + (1,)), dtype="float32")
    y_out = tf.constant(lbl.reshape((1,) + shape + (1,)), dtype="float32")

    weights = np.array(
        [
            [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
            [0.00501119, 0.05221780, 0.11405416, 0.05221780, 0.00501119],
            [0.01094545, 0.11405416, 0.24911720, 0.11405416, 0.01094545],
            [0.00501119, 0.05221780, 0.11405416, 0.05221780, 0.00501119],
            [0.00048091, 0.00501119, 0.01094545, 0.00501119, 0.00048091],
        ],
        dtype=np.float32,
    )
    kernel = tf.constant(weights.reshape(weights.shape + (1, 1)), dtype="float32")

    n = 24
    cnv = lbl.copy()
    for _ in range(n):
        cnv = sp.ndimage.convolve(cnv, weights, mode="reflect")
    cnv.round(2)[100:-100, 100:-100]

    x = input = tf.keras.Input(shape + (1,), name="input", dtype="float32")
    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
    x = tf.keras.layers.Conv2D(3, 3, padding="VALID")(x)
    x = tf.keras.layers.Conv2D(1, 1, name="final_conv")(x)
    x = tf.keras.layers.Activation("sigmoid", dtype="float32", name="output")(x)
    x = tf.pad(x, [[0, 0], [2 * n, 2 * n], [2 * n, 2 * n], [0, 0]], mode="REFLECT")
    for i in range(n):
        x = tf.nn.conv2d(x, kernel, strides=1, padding="VALID")
    output = x
    model = tf.keras.Model(inputs=input, outputs=output)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_in, y_out, epochs=1000, verbose=0)

    res = model.predict(x_in)
    res.round(2)[0, 95:-105, 100:-100, 0]
    res.round(2)[0, :, :, 0]
    arr.mean(), arr.std()
    cnv.mean(), cnv.std()
    res.mean(), res.std()

    np.abs((res - cnv).round(3)).max()
