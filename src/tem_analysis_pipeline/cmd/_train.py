import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import tensorflow as tf

from ..model.utils import to_numpy_or_python_type

if TYPE_CHECKING:
    import keras
    from keras import Model
    from keras.src.callbacks import ModelCheckpoint
    from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
else:
    import tensorflow.keras as keras
    from tensorflow.data import Dataset
    from tensorflow.keras import Model
    from tensorflow.keras.callbacks import ModelCheckpoint


keras.config.set_dtype_policy("float32")


class PersistentBestModelCheckpoint(ModelCheckpoint):
    best: dict
    json_path: Path

    def __init__(self, dirpath: str | Path, **kwargs):
        if "save_best_only" in kwargs.keys() and not kwargs.get("save_best_only"):
            print(
                "Warning: The PersistentBestModelCheckpoint is specifically for saving best models accross runs."
            )
            print("         Will ignore the provided `save_best_only=False` option.")
        kwargs.update(dict(save_best_only=True))

        dirpath = Path(dirpath)

        super().__init__(dirpath.with_suffix(".keras"), **kwargs)

        assert self.save_freq == "epoch"

        if dirpath.exists():
            assert dirpath.is_dir()
        else:
            dirpath.mkdir(parents=True)

        self.json_path = dirpath / "best_logs.json"
        if self.json_path.exists():
            with open(self.json_path, "r") as file:
                best_logs: dict = json.load(file)

            self.best = best_logs.get(self.monitor)

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        try:
            logs = logs or {}
            logs = to_numpy_or_python_type(logs)
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                with open(self.json_path, "w") as file:
                    json.dump(logs, file)

        finally:
            super().on_epoch_end(epoch, logs)


def _is_gzipped(filepath: str | Path) -> bool:
    with open(filepath, "rb") as file:
        return file.read(2) == b"\x1f\x8b"


def get_dataset(
    dataset_name: str,
    split: Literal["tra_val", "tst"],
    organelle: Literal["cell", "mitochondria", "nucleus"],
    tile_shape: tuple[int, int],
    window_shape: tuple[int, int],
    batch_size: int,
    fold_n: int = 1,
    total_folds: int = 1,
    fraction_of_empty_to_keep: float = 1.0,
    shuffle: bool = True,
    buffer_size: int = 5_000,
    cache: bool = False,
    data_dirpath: str | Path = "data",
) -> Dataset | tuple[Dataset, Dataset | None]:
    from ..model.utils import (
        crop_labels_to_shape,
        keep_fraction_of_empty_labels,
        random_flip_and_rotation,
        random_image_adjust,
        read_tfrecord,
        set_tile_shape,
    )

    if not 0 < fold_n <= total_folds:
        raise ValueError(f"fold_n must be between 1 and {total_folds}. Got {fold_n}.")

    data_dirpath = Path(data_dirpath)
    dataset_dirpath = data_dirpath / dataset_name
    params = dict(num_parallel_calls=tf.data.AUTOTUNE)

    tfrecords_dirpath = dataset_dirpath / split / organelle / "tfrecords"
    files = tf.data.Dataset.list_files(f"{tfrecords_dirpath}/*.tfrecord", shuffle=False)

    first_filepath = tfrecords_dirpath.glob("*.tfrecord").__next__()
    compression_type = "GZIP" if _is_gzipped(first_filepath) else None

    def _load_tfrecord_dataset(filepath: str) -> tf.data.TFRecordDataset:
        return tf.data.TFRecordDataset(filepath, compression_type=compression_type)

    dataset = files.interleave(_load_tfrecord_dataset, deterministic=True, **params)
    dataset = dataset.map(read_tfrecord, **params)
    dataset = dataset.map(set_tile_shape(tile_shape), **params)
    if cache:
        dataset = dataset.cache()

    _label_crop_fn = crop_labels_to_shape(window_shape)

    if split == "tra_val":
        if fraction_of_empty_to_keep < 1.0:
            dataset = dataset.filter(
                keep_fraction_of_empty_labels(fraction_of_empty_to_keep)
            )

        dataset = dataset.enumerate()

        def is_validation(index, data):
            return total_folds > 1 and index % total_folds == fold_n - 1

        def is_training(index, data):
            return total_folds == 1 or index % total_folds != fold_n - 1

        training_dataset = dataset.filter(is_training).map(lambda i, d: d)
        validation_dataset = dataset.filter(is_validation).map(lambda i, d: d)

        num_training_elements = training_dataset.reduce(
            tf.constant(0, dtype=tf.int64), lambda x, _: x + 1
        ).numpy()
        print(f"Number of training elements: {num_training_elements}", flush=True)

        num_validation_elements = validation_dataset.reduce(
            tf.constant(0, dtype=tf.int64), lambda x, _: x + 1
        ).numpy()
        print(f"Number of validation elements: {num_validation_elements}", flush=True)

        training_dataset = training_dataset.map(random_flip_and_rotation, **params)
        training_dataset = training_dataset.map(random_image_adjust, **params)
        if shuffle:
            buffer_size = num_training_elements if buffer_size < 0 else buffer_size
            print(f"Shuffling training data. Buffer size: {buffer_size}", flush=True)
            training_dataset = training_dataset.shuffle(buffer_size=buffer_size)

        training_dataset = training_dataset.map(_label_crop_fn, **params)
        training_dataset = training_dataset.repeat()
        training_dataset = training_dataset.batch(batch_size)
        training_dataset = training_dataset.prefetch(tf.data.AUTOTUNE)

        if num_validation_elements > 0:
            validation_dataset = validation_dataset.map(_label_crop_fn, **params)
            validation_dataset = validation_dataset.repeat()
            validation_dataset = validation_dataset.batch(batch_size)
            validation_dataset = validation_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            validation_dataset = None

        return (
            num_training_elements,
            training_dataset,
            num_validation_elements,
            validation_dataset,
        )

    elif split == "tst":
        dataset = dataset.map(_label_crop_fn, **params)
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset

    else:
        raise ValueError(f"Unknown split: {split}")


def train(
    dataset_name: str,
    organelle: Literal["cell", "mitochondria", "nucleus"] = "mitochondria",
    fold_n: int = 1,
    total_folds: int = 1,
    shuffle_training: bool = False,
    buffer_size: int = 5_000,
    batch_size: int | None = None,
    cache: bool = False,
    n_epochs_per_run: int = 1200,
    data_dirpath: str | Path = "data",
) -> None:
    """Train a U-Net model for semantic segmentation of TEM images."""
    from ..model._unet import make_unet
    from ..model.config import config
    from ..model.custom_objects import custom_objects
    from ..model.losses import MyWeightedBinaryCrossEntropy
    from ..model.metrics import (
        MyF1Score,
        MyF2Score,
    )

    if total_folds > 1:
        working_dir = Path(f"models/{total_folds}-fold_cross_validation")
    else:
        working_dir = Path("models/single_fold")
    working_dir = working_dir / dataset_name / organelle / f"kf{fold_n:02d}"
    working_dir.mkdir(parents=True, exist_ok=True)

    if organelle not in config.keys():
        print(
            f"Error: Could not find the organelle name {organelle} in the config file. Exiting..."
        )
        exit()

    loss_pos_weight = 2
    params = config[organelle]
    tile_shape = params["tile_shape"]
    layer_depth = params["layer_depth"]
    window_shape = params["window_shape"]
    batch_size = batch_size or params["batch_size"]
    filters_root = params["filters_root"]
    fraction_of_empty_to_keep = params["fraction_of_empty_to_keep"]

    n_train, training_dataset, n_val, validation_dataset = get_dataset(
        dataset_name,
        "tra_val",
        organelle,
        tile_shape,
        window_shape,
        batch_size,
        fold_n=fold_n,
        total_folds=total_folds,
        fraction_of_empty_to_keep=fraction_of_empty_to_keep,
        shuffle=shuffle_training,
        buffer_size=buffer_size,
        cache=cache,
        data_dirpath=data_dirpath,
    )
    steps_per_epoch = n_train // batch_size
    validation_steps = n_val // batch_size

    model: Model
    if os.path.exists(f"{working_dir}/ckpt/last.keras"):
        with open(f"{working_dir}/logs/metrics.tsv") as file:
            initial_epoch = sum(1 for line in file) - 1
        model = keras.models.load_model(
            f"{working_dir}/ckpt/last.keras", custom_objects=custom_objects
        )
    else:
        initial_epoch = 0
        # building the model
        model = make_unet(
            tile_shape, channels=1, layer_depth=layer_depth, filters_root=filters_root
        )

        loss = MyWeightedBinaryCrossEntropy(pos_weight=loss_pos_weight)
        f1_score_metric = MyF1Score(name="f1_score", threshold=0.5)
        f2_score_metric = MyF2Score(name="f2_score", threshold=0.5)

        model.compile(
            loss=loss,
            optimizer="adamw",
            metrics=[
                "Recall",
                f1_score_metric,
                f2_score_metric,
            ],
            jit_compile=False,
        )

    total_epochs = n_epochs_per_run * (initial_epoch // n_epochs_per_run + 1)

    history = model.fit(
        training_dataset,
        validation_data=validation_dataset,
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            keras.callbacks.CSVLogger(
                f"{working_dir}/logs/metrics.tsv", separator="\t", append=True
            ),
            keras.callbacks.TensorBoard(f"{working_dir}/logs", profile_batch=0),
            keras.callbacks.ModelCheckpoint(f"{working_dir}/ckpt/last.keras"),
            *(
                [
                    PersistentBestModelCheckpoint(
                        f"{working_dir}/ckpt/best_loss",
                        save_best_only=True,
                        monitor="val_loss",
                        verbose=1,
                    )
                ]
                if total_folds > 1
                else []
            ),
        ],
        verbose=1,
    )
    final_epoch = initial_epoch + len(history.history["loss"])

    test_dataset = get_dataset(
        dataset_name,
        "tst",
        organelle,
        tile_shape,
        window_shape,
        batch_size=8,
        data_dirpath=data_dirpath,
    )
    thresholds = np.arange(0, 1, 0.005).tolist()
    metrics = [
        keras.metrics.TruePositives(thresholds=thresholds),
        keras.metrics.FalsePositives(thresholds=thresholds),
        keras.metrics.TrueNegatives(thresholds=thresholds),
        keras.metrics.FalseNegatives(thresholds=thresholds),
    ]
    model.compile(metrics=metrics)
    result = model.evaluate(test_dataset, verbose=2)[1:]
    result = [r.numpy().astype(int).tolist() for r in result]

    json_path = working_dir / "evaluation" / f"{final_epoch:05d}_evaluation.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as file:
        json.dump(result, file)
