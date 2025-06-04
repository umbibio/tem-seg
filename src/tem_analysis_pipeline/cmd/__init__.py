"""Command-line interface for TEM analysis pipeline."""

from pathlib import Path
from typing import List, Optional, Annotated

import typer
from typer import Option, Argument


# Create the main app
app = typer.Typer(
    help="TEM Analysis Pipeline - Tools for training models and computing predictions on TEM images"
)

@app.command("train")
def train_command(
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
    from ._train import train

    train(working_dir, organelle, k_fold, n_epochs_per_run)


@app.command("predict")
def predict_command(
    filepaths: Annotated[List[Path], Argument(help="Paths to the image files")],
    model_version: Annotated[
        str, Option("--model-version", help="Version of the model to use")
    ] = "Mixture",
    organelle: Annotated[
        str, Option("--organelle", help="Target organelle for prediction")
    ] = "mitochondria",
    trg_scale: Annotated[
        float, Option("--trg-scale", help="Target scale for prediction")
    ] = 0.0075,
    force_prediction: Annotated[
        bool,
        Option("--force-prediction", help="Force prediction even if output exists"),
    ] = False,
    models_folder: Annotated[
        Optional[Path],
        Option("--models-folder", help="Optional folder containing models"),
    ] = None,
    use_ensemble: Annotated[
        bool, Option("--use-ensemble", help="Use ensemble model")
    ] = False,
) -> None:
    """Compute predictions for the given image files using the specified model."""
    from ._compute_prediction import compute_prediction

    compute_prediction(
        filepaths=filepaths,
        model_version=model_version,
        organelle=organelle,
        trg_scale=trg_scale,
        force_prediction=force_prediction,
        models_folder=models_folder,
        use_ensemble=use_ensemble,
    )


preprocess_app = typer.Typer(help="Preprocess slides and masks for training")
app.add_typer(preprocess_app, name="preprocess")

@preprocess_app.command("tfrecords")
def preprocess_tfrecords(
    slides_dirpath: Annotated[
        Path, Argument(help="Directory containing slide images")
    ],
    masks_dirpath: Annotated[
        Path, Argument(help="Directory containing mask images")
    ],
    organelle: Annotated[
        str, Option("--organelle", "-o", help="Target organelle for preprocessing")
    ],
    output_dirpath: Annotated[
        Path, Option("--output-dirpath", "-d", help="Output directory path")
    ],
    slide_format: Annotated[
        str, Option("--slide-format", "-f", help="Format of slide images")
    ] = "tif",
) -> None:
    """Preprocess slides and masks into TFRecords format for training."""
    from ._preprocess import make_tfrecords

    make_tfrecords(
        slides_dirpath=slides_dirpath,
        masks_dirpath=masks_dirpath,
        organelle=organelle,
        slide_format=slide_format,
        output_dirpath=output_dirpath,
    )


if __name__ == "__main__":
    app()
