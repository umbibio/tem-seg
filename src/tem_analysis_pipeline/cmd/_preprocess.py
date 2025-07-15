import gc
import os
from pathlib import Path

import PIL.Image
import PIL.TiffImagePlugin

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from ..calibration import NoScaleError, get_calibration

PIL.Image.MAX_IMAGE_PIXELS = 10_000_000_000


def load_image(img_filepath: str | Path) -> PIL.Image.Image:
    img_filepath = Path(img_filepath)

    img = PIL.Image.open(img_filepath)
    if img.mode != "L":
        x = np.array(img)
        x = ((x / np.iinfo(x.dtype).max) * 255).round().astype(np.uint8)
        img = PIL.Image.fromarray(x)

    return img


def load_mask(msk_filepath: str | Path) -> PIL.Image.Image:
    msk_filepath = Path(msk_filepath)

    msk = PIL.Image.open(msk_filepath)
    n_channels = len(msk.getbands())
    assert n_channels == 1, f"Mask must have a single channel, got {n_channels}"

    if msk.mode == "P":
        arr = np.array(msk).astype(float)
        assert arr.max() == 1
        msk = PIL.Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    return msk


def save_fixed_scale_sample(
    img_filepath: str | Path,
    msk_filepath: str | Path,
    organelle: str,
    target_scale: float,
    output_dirpath: str | Path,
    i_size: int,
    o_size: int,
) -> None:
    assert target_scale > 0.0

    stride = o_size // 2

    img_filepath = Path(img_filepath)
    msk_filepath = Path(msk_filepath)
    spl_filepath = (
        output_dirpath / organelle / "tfrecords" / f"{img_filepath.stem}.tfrecord"
    )
    if spl_filepath.exists():
        print(f"Skipping. {spl_filepath} already exists.")
        return

    # load image
    img = load_image(img_filepath)

    # load mask
    msk = load_mask(msk_filepath)

    assert img.size == msk.size, "Image and mask must have the same size"

    try:
        img_scale = get_calibration(img)
    except NoScaleError:
        print(f"No scale found for {img_filepath}. Skipping.")
        return

    scale_ratio = target_scale / img_scale
    img_width, img_height = img.size

    trg_width = int(np.round(img_width / scale_ratio))
    trg_height = int(np.round(img_height / scale_ratio))
    trg_size = (trg_width, trg_height)

    img = img.resize(trg_size, resample=PIL.Image.Resampling.LANCZOS)
    img = img.crop(
        (
            -i_size // 2,
            -i_size // 2,
            trg_width + i_size // 2,
            trg_height + i_size // 2,
        )
    )
    img_arr = np.array(img)

    msk = msk.resize(trg_size, resample=PIL.Image.Resampling.NEAREST)
    msk = msk.crop(
        (
            -i_size // 2,
            -i_size // 2,
            trg_width + i_size // 2,
            trg_height + i_size // 2,
        )
    )
    msk_arr = np.array(msk)

    img_arr_swv = np.lib.stride_tricks.sliding_window_view(img_arr, (i_size, i_size))[
        ::stride, ::stride
    ]
    msk_arr_swv = np.lib.stride_tricks.sliding_window_view(msk_arr, (i_size, i_size))[
        ::stride, ::stride
    ]

    sampls = img_arr_swv.reshape((-1, i_size, i_size))
    labels = msk_arr_swv.reshape((-1, i_size, i_size))

    sampls = (sampls / 255).astype(np.float16)
    labels = (labels / 255).astype(np.float16)

    del img, msk, img_arr, msk_arr, img_arr_swv, msk_arr_swv
    gc.collect()

    def _data_generator():
        for i in range(sampls.shape[0]):
            yield (sampls[i], labels[i])

    dataset = tf.data.Dataset.from_generator(
        _data_generator,
        output_signature=(
            tf.TensorSpec(shape=(i_size, i_size), dtype=tf.float16),
            tf.TensorSpec(shape=(i_size, i_size), dtype=tf.float16),
        ),
    )

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    spl_filepath.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(spl_filepath.as_posix(), options=options) as file_writer:
        for record in dataset:
            example = _make_example(record)
            record_bytes = example.SerializeToString()
            file_writer.write(record_bytes)


def _make_bytes_feature(tile):
    tile = tf.expand_dims(tile, 2)
    tile_serialized = tf.io.serialize_tensor(tile)
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tile_serialized.numpy()])
    )


def _make_example(record: tuple[tf.Tensor, tf.Tensor]) -> tf.train.Example:
    img_tile, msk_tile = record
    img_tile_feature_of_bytes = _make_bytes_feature(img_tile)
    msk_tile_feature_of_bytes = _make_bytes_feature(msk_tile)

    features_for_example = {
        "img_tile": img_tile_feature_of_bytes,
        "msk_tile": msk_tile_feature_of_bytes,
    }

    return tf.train.Example(features=tf.train.Features(feature=features_for_example))


def cast(v):
    # https://github.com/python-pillow/Pillow/issues/6199#issuecomment-1214854558
    if isinstance(v, PIL.TiffImagePlugin.IFDRational):
        return float(v)
    elif isinstance(v, tuple):
        return tuple(cast(t) for t in v)
    elif isinstance(v, bytes):
        return v.decode(errors="replace")
    elif isinstance(v, dict):
        for kk, vv in v.items():
            v[kk] = cast(vv)
        return v
    else:
        return v


def cast_dict(d):
    # https://github.com/python-pillow/Pillow/issues/6199#issuecomment-1214854558
    for k, v in d.items():
        d[k] = cast(v)
    return d


def make_tfrecords(
    dataset_name: str,
    slides_dirpath: Path,
    masks_dirpath: Path,
    organelle: str,
    slide_format: str = "tif",
    test_size: float | int = 0,
    random_state: int = 42,
    data_dirpath: str | Path = "data",
) -> None:
    from ..model.config import config

    data_dirpath = Path(data_dirpath)
    output_dirpath = data_dirpath / dataset_name

    i_size = config[organelle]["tile_shape"][0]
    o_size = config[organelle]["window_shape"][0]
    target_scale = config[organelle]["target_scale"]

    print("\ngenerating fixed scale samples")

    slides = slides_dirpath.glob(f"*.{slide_format}")

    valid_pairs: list[tuple[Path, Path]] = []
    for slide in sorted(slides):
        slide_name = slide.stem
        mask = masks_dirpath / f"{slide_name}.png"
        if not mask.exists():
            print(f"Didn't find mask for {slide_name}. Skipping.", flush=True)
            continue

        valid_pairs.append((slide, mask))

    train_pairs: list[tuple[Path, Path]]
    test_pairs: list[tuple[Path, Path]]
    if test_size > 0:
        train_pairs, test_pairs = train_test_split(
            valid_pairs, test_size=test_size, random_state=random_state
        )
    elif test_size == 0:
        train_pairs = valid_pairs
        test_pairs = []
    else:
        train_pairs = []
        test_pairs = valid_pairs

    print(f"\n{len(train_pairs)} training samples")
    print(f"{len(test_pairs)} test samples")

    i = 0
    for slide, mask in train_pairs:
        i += 1
        print(f"training slide {i: 4d}:", slide.name, flush=True, end="\t")
        save_fixed_scale_sample(
            slide,
            mask,
            organelle=organelle,
            target_scale=target_scale,
            output_dirpath=output_dirpath / "tra_val",
            i_size=i_size,
            o_size=o_size,
        )

    i = 0
    for slide, mask in test_pairs:
        i += 1
        print(f"testing slide {i: 4d}:", slide.name, flush=True, end="\t")
        save_fixed_scale_sample(
            slide,
            mask,
            organelle=organelle,
            target_scale=target_scale,
            output_dirpath=output_dirpath / "tst",
            i_size=i_size,
            o_size=o_size,
        )
