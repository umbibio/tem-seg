from pathlib import Path
from typing import Literal, Annotated
import json
import os

import typer
from typer import Option, Argument
import keras
import numpy as np
import tensorflow as tf

from ..model.utils import to_numpy_or_python_type


keras.config.set_dtype_policy("float32")


class PersistentBestModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, **kwargs):
        if "save_best_only" in kwargs.keys() and not kwargs.get("save_best_only"):
            print(
                "Warning: The PersistentBestModelCheckpoint is specifically for saving best models accross runs."
            )
            print("         Will ignore the provided `save_best_only=False` option.")
        kwargs.update(dict(save_best_only=True))

        super().__init__(filepath, **kwargs)

        assert self.save_freq == "epoch"

        if os.path.exists(filepath):
            assert os.path.isdir(filepath)
        else:
            os.makedirs(filepath)

        self.json_path = os.path.join(filepath, "best_logs.json")
        if os.path.exists(self.json_path):
            with open(self.json_path, "r") as file:
                best_logs = json.load(file)

            self.best = best_logs.get(self.monitor)

    def on_epoch_end(self, epoch, logs=None):
        try:
            logs = logs or {}
            logs = to_numpy_or_python_type(logs)
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                with open(self.json_path, "w") as file:
                    json.dump(logs, file)

        finally:
            super().on_epoch_end(epoch, logs)


def get_dataset(
    tile_shape: tuple[int, int],
    window_shape: tuple[int, int],
    fold_dir: Path,
    split: Literal["tra", "val", "tst"],
    fraction_of_empty_to_keep: float = 1.0,
    shuffle: bool = True,
):
    from ..model.utils import crop_labels_to_shape
    from ..model.utils import (
        read_tfrecord,
        set_tile_shape,
        random_flip_and_rotation,
        random_image_adjust,
        keep_fraction_of_empty_labels,
    )

    files = tf.data.Dataset.list_files(f"{fold_dir}/{split}/*.tfrecord")
    dataset = files.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True
    )
    dataset = dataset.map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        set_tile_shape(tile_shape), num_parallel_calls=tf.data.AUTOTUNE
    )

    if fraction_of_empty_to_keep < 1.0:
        dataset = dataset.filter(
            keep_fraction_of_empty_labels(fraction_of_empty_to_keep)
        )

    if split in ["tra", "val"]:
        dataset = dataset.cache()
        dataset = dataset.map(
            random_flip_and_rotation, num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.map(random_image_adjust, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=32, reshuffle_each_iteration=True)

    dataset = dataset.map(
        crop_labels_to_shape(window_shape), num_parallel_calls=tf.data.AUTOTUNE
    )
    return dataset


def train(
    working_dir: Path,
    organelle: Literal["cell", "mitochondria", "nucleus"] = "mitochondria",
    k_fold: int = 1,
    n_epochs_per_run: int = 1200,
) -> None:
    """Train a U-Net model for semantic segmentation of TEM images."""
    from ..model.losses import MyWeightedBinaryCrossEntropy
    from ..model.metrics import (
        MyMeanDSC,
        MyMeanIoU,
        MyJaccardIndex,
        MyF1Score,
        MyF2Score,
    )
    from ..model.custom_objects import custom_objects
    from ..model._unet import make_unet

    from ..model.config import config

    working_dir = Path(working_dir)
    fold_dir = working_dir / f"kf{k_fold}"

    if organelle not in config.keys():
        print(
            "Error: Could not find the organelle name from the current working directory. Exiting..."
        )
        exit()

    loss_pos_weight = 2
    params = config[organelle]
    tile_shape = params["tile_shape"]
    layer_depth = params["layer_depth"]
    window_shape = params["window_shape"]
    batch_size = params["batch_size"]
    filters_root = params["filters_root"]
    fraction_of_empty_to_keep = params["fraction_of_empty_to_keep"]

    train_dataset = get_dataset(
        tile_shape,
        window_shape,
        fold_dir,
        split="tra",
        fraction_of_empty_to_keep=fraction_of_empty_to_keep,
    )
    validation_dataset = get_dataset(
        tile_shape,
        window_shape,
        fold_dir,
        split="val",
        fraction_of_empty_to_keep=fraction_of_empty_to_keep,
    )

    if os.path.exists(f"{fold_dir}/ckpt/last/saved_model.pb"):
        with open(f"{fold_dir}/logs/metrics.tsv") as file:
            initial_epoch = sum(1 for line in file) - 1
        model = tf.keras.models.load_model("ckpt/last", custom_objects=custom_objects)
    else:
        initial_epoch = 0
        # building the model
        model = make_unet(
            tile_shape, channels=1, layer_depth=layer_depth, filters_root=filters_root
        )

        loss = MyWeightedBinaryCrossEntropy(pos_weight=loss_pos_weight)
        mean_iou_metric = MyMeanIoU(name="mean_iou", threshold=0.5)
        mean_dsc_metric = MyMeanDSC(name="dice_coefficient", threshold=0.5)
        jc_index_metric = MyJaccardIndex(name="jaccard_index", threshold=0.5)
        f1_score_metric = MyF1Score(name="f1_score", threshold=0.5)
        f2_score_metric = MyF2Score(name="f2_score", threshold=0.5)

        model.compile(
            loss=loss,
            optimizer="adam",
            metrics=[
                mean_iou_metric,
                mean_dsc_metric,
                "Recall",
                jc_index_metric,
                f1_score_metric,
                f2_score_metric,
            ],
            jit_compile=False,
        )

    total_epochs = n_epochs_per_run * (initial_epoch // n_epochs_per_run + 1)

    history = model.fit(
        train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
        validation_data=validation_dataset.batch(
            batch_size, drop_remainder=True
        ).prefetch(tf.data.AUTOTUNE),
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=[
            tf.keras.callbacks.CSVLogger(
                f"{fold_dir}/logs/metrics.tsv", separator="\t", append=True
            ),
            tf.keras.callbacks.TensorBoard(f"{fold_dir}/logs", profile_batch=0),
            tf.keras.callbacks.ModelCheckpoint(f"{fold_dir}/ckpt/last"),
            PersistentBestModelCheckpoint(
                f"{fold_dir}/ckpt/best_loss",
                save_best_only=True,
                monitor="val_loss",
                verbose=1,
            ),
            PersistentBestModelCheckpoint(
                f"{fold_dir}/ckpt/best_dice",
                save_best_only=True,
                monitor="val_dice_coefficient",
                mode="max",
                verbose=1,
            ),
            PersistentBestModelCheckpoint(
                f"{fold_dir}/ckpt/best_f2",
                save_best_only=True,
                monitor="val_f2_score",
                mode="max",
                verbose=1,
            ),
        ],
        verbose=2,
        workers=12,
    )
    final_epoch = initial_epoch + len(history.history["loss"])
    model.save(f"{fold_dir}/ckpt/intermediate/{final_epoch:05d}")

    test_dataset = get_dataset(tile_shape, window_shape, fold_dir, split="tst")
    thresholds = np.arange(0, 1, 0.005).tolist()
    metrics = [
        tf.keras.metrics.TruePositives(thresholds=thresholds),
        tf.keras.metrics.FalsePositives(thresholds=thresholds),
        tf.keras.metrics.TrueNegatives(thresholds=thresholds),
        tf.keras.metrics.FalseNegatives(thresholds=thresholds),
    ]
    model.compile(metrics=metrics)
    result = model.evaluate(test_dataset.batch(8), verbose=2)[1:]
    result = [r.astype(int).tolist() for r in result]

    json_path = os.path.join(
        fold_dir, "evaluation", f"{final_epoch:05d}_evaluation.json"
    )
    with open(json_path, "w") as file:
        json.dump(result, file)


app = typer.Typer(help="Train a U-Net model for semantic segmentation of TEM images")


@app.command()
def main(
    working_dir: Annotated[
        Path, Argument(help="Working directory containing the dataset")
    ],
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for segmentation")
    ] = "mitochondria",
    k_fold: Annotated[
        int, Option("--k-fold", "-k", help="K-fold cross-validation fold number")
    ] = 1,
    n_epochs_per_run: Annotated[
        int,
        Option("--n-epochs-per-run", "-e", help="Number of epochs per training run"),
    ] = 1200,
) -> None:
    """Train a U-Net model for semantic segmentation of TEM images."""
    train(working_dir, organelle, k_fold, n_epochs_per_run)


if __name__ == "__main__":
    app()
