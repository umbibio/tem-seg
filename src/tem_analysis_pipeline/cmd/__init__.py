"""Command-line interface for TEM analysis pipeline."""

from pathlib import Path
from typing import Annotated, List

import typer
from typer import Argument, Option

# Create the main app
app = typer.Typer(
    help="TEM Analysis Pipeline - Tools for training models and computing predictions on TEM images"
)


@app.command("train")
def train_command(
    dataset_name: Annotated[str, Argument(help="Name of the dataset to train on")],
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for segmentation")
    ] = "mitochondria",
    fold_n: Annotated[int, Option("--fold-n", "-f", help="Fold number")] = 1,
    total_folds: Annotated[
        int, Option("--total-folds", "-k", help="Total number of folds")
    ] = 1,
    shuffle_training: Annotated[
        bool, Option("--shuffle-training", "-s", help="Shuffle training data")
    ] = False,
    batch_size: Annotated[
        int | None, Option("--batch-size", "-b", help="Batch size")
    ] = None,
    n_epochs_per_run: Annotated[
        int,
        Option("--n-epochs-per-run", "-e", help="Number of epochs per training run"),
    ] = 1200,
    model_type: Annotated[
        str, Option("--model-type", "-m", help="Type of model to train ('unet' or 'unet++')")
    ] = "unet",
    deep_supervision: Annotated[
        bool, Option("--deep-supervision", "-d", help="Use deep supervision in UNet++")
    ] = False,
) -> None:
    """Train a U-Net model for semantic segmentation of TEM images."""
    from ._train import train

    train(
        dataset_name,
        organelle,
        fold_n,
        total_folds,
        shuffle_training,
        batch_size,
        n_epochs_per_run,
        model_type,
        deep_supervision,
    )


@app.command("predict")
def predict_command(
    filepaths: Annotated[List[Path], Argument(help="Paths to the image files")],
    model_version: Annotated[
        str, Option("--model-version", "-v", help="Version of the model to use")
    ] = "Mixture",
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for prediction")
    ] = "mitochondria",
    force_prediction: Annotated[
        bool,
        Option(
            "--force-prediction", "-f", help="Force prediction even if output exists"
        ),
    ] = False,
    models_folder: Annotated[
        Path | None,
        Option("--models-folder", "-m", help="Optional folder containing models"),
    ] = None,
    use_ensemble: Annotated[
        bool, Option("--use-ensemble", "-e", help="Use ensemble model")
    ] = False,
    checkpoint: Annotated[
        str, Option("--checkpoint", "-c", help="Checkpoint to use")
    ] = "last",
    cross_validation_kfolds: Annotated[
        int | None,
        Option("--cross-validation-kfolds", "-k", help="Cross-validation k-folds"),
    ] = None,
) -> None:
    """Compute predictions for the given image files using the specified model."""
    from ._compute_prediction import compute_prediction

    compute_prediction(
        filepaths=filepaths,
        model_version=model_version,
        organelle=organelle,
        force_prediction=force_prediction,
        models_folder=models_folder,
        use_ensemble=use_ensemble,
        checkpoint=checkpoint,
        cross_validation_kfolds=cross_validation_kfolds,
    )


@app.command("analyze")
def analyze_command(
    study_name: Annotated[str, Argument(help="Name of the study to analyze")],
    model_name: Annotated[
        str,
        Option(
            "--model-name",
            "-n",
            help="Name of the model to use. E.g. Mixture or Mixture_k5",
        ),
    ] = "Mixture",
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for analysis")
    ] = "mitochondria",
    redo_analysis: Annotated[
        bool,
        Option("--redo-analysis", "-r", help="Redo analysis even if output exists"),
    ] = False,
    force_convolution: Annotated[
        bool,
        Option(
            "--force-convolution",
            "-c",
            help="Force convolution even if soft prediction exists",
        ),
    ] = False,
    n_jobs: Annotated[
        int, Option("--n-jobs", "-j", help="Number of parallel jobs")
    ] = 1,
) -> None:
    """Analyze predictions for the given image files using the specified model."""
    from ._compute_analysis import compute_analysis

    compute_analysis(
        study_name=study_name,
        model_name=model_name,
        organelle=organelle,
        redo_analysis=redo_analysis,
        force_convolution=force_convolution,
        n_jobs=n_jobs,
    )


@app.command("consolidate")
def consolidate_command(
    study_name: Annotated[str, Argument(help="Name of the study to consolidate results for")],
    model_name: Annotated[
        str,
        Option(
            "--model-name",
            "-n",
            help="Name of the model used for predictions",
        ),
    ] = "Mixture",
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for consolidation")
    ] = "mitochondria",
    output_file: Annotated[
        str,
        Option(
            "--output-file",
            "-f",
            help="Path to save the consolidated CSV file",
        ),
    ] = None,
) -> None:
    """Consolidate all CSV analysis results for a study into a single file."""
    from ..analysis._consolidate import consolidate_study_results

    if output_file is None:
        output_file = f"results/{study_name}_{model_name}_{organelle}_consolidated.csv"

    consolidate_study_results(
        study_name=study_name,
        model_name=model_name,
        organelle=organelle,
        output_file=output_file,
    )


preprocess_app = typer.Typer(help="Preprocess slides and masks for training")
app.add_typer(preprocess_app, name="preprocess")


@preprocess_app.command("tfrecords")
def preprocess_tfrecords(
    dataset_name: Annotated[str, Argument(help="Name of the dataset")],
    slides_dirpath: Annotated[Path, Argument(help="Directory containing slide images")],
    masks_dirpath: Annotated[Path, Argument(help="Directory containing mask images")],
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for preprocessing")
    ],
    test_size: Annotated[
        float, Option("--test-size", "-t", help="Fraction of data to use for testing")
    ] = 0.0,
    slide_format: Annotated[
        str, Option("--slide-format", "-f", help="Format of slide images")
    ] = "tif",
    random_state: Annotated[
        int, Option("--random-state", "-r", help="Random state for splitting data")
    ] = 42,
) -> None:
    """Preprocess slides and masks into TFRecords format for training."""
    import tensorflow as tf

    from ._preprocess import make_tfrecords

    with tf.device("/CPU:0"):
        make_tfrecords(
            dataset_name=dataset_name,
            slides_dirpath=slides_dirpath,
            masks_dirpath=masks_dirpath,
            organelle=organelle,
            slide_format=slide_format,
            test_size=test_size,
            random_state=random_state,
        )


if __name__ == "__main__":
    app()
