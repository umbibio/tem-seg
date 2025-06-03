from pathlib import Path
import re
from typing import List, Optional, Union, Annotated

import typer
from typer import Option, Argument

from tem_analysis_pipeline import prediction_tools
from tem_analysis_pipeline.calibration import (
    NoScaleError,
    NoScaleNumberError,
    NoScaleUnitError,
)


def compute_prediction(
    filepaths: List[Path],
    model_version: str,
    organelle: str,
    trg_scale: float,
    force_prediction: bool = False,
    models_folder: Optional[Union[str, Path]] = None,
    use_ensemble: bool = False,
) -> None:
    """
    Compute predictions for the given image files using the specified model.

    Args:
        filepaths: List of paths to the image files to process
        model_version: Version of the model to use
        organelle: Target organelle for the prediction
        trg_scale: Target scale for the prediction
        force_prediction: Whether to force prediction even if output file exists
        models_folder: Optional folder containing the models
        use_ensemble: Whether to use ensemble model
    """
    predictions_basedir = Path("prediction")
    if use_ensemble:
        predictions_basedir /= model_version + "_ensemble"
    else:
        predictions_basedir /= model_version
    predictions_basedir /= organelle

    prediction_tools.select_model_version(
        model_version,
        models_folder=models_folder,
        use_ensemble=use_ensemble,
    )
    model = None

    for img_filepath in filepaths:
        assert img_filepath.exists(), f"Image file {img_filepath} does not exist"

        output_basedir = img_filepath.parent.joinpath(predictions_basedir)

        prd_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{organelle}.png", img_filepath.name)
        )
        print(img_filepath, flush=True)

        img = prediction_tools.load_image(img_filepath.as_posix())

        if not prd_filepath.exists() or force_prediction:
            prd_filepath.parent.mkdir(parents=True, exist_ok=True)

            if model is None:
                model = prediction_tools.get_organelle_ensemble_model(organelle)

            try:
                prd = prediction_tools.image_prediction(img, model, trg_scale=trg_scale)
                prd.save(prd_filepath)

            except NoScaleError as e:
                print(
                    f"Warning: no scale could be read for image {img_filepath}",
                    flush=True,
                )
                print(e, flush=True)
                continue

            except NoScaleNumberError as e:
                print(
                    f"Warning: no scale number could be read for image {img_filepath}",
                    flush=True,
                )
                print(e, flush=True)
                continue

            except NoScaleUnitError as e:
                print(
                    f"Warning: no scale unit could be read for image {img_filepath}",
                    flush=True,
                )
                print(e, flush=True)
                continue

            except AssertionError as e:
                print(
                    f"Warning: prediction failed for image {img_filepath}", flush=True
                )
                print(e, flush=True)
                continue

        print(prd_filepath, flush=True)


app = typer.Typer(help="Compute predictions for TEM images")


@app.command()
def main(
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
    """Command line interface for compute_prediction function."""
    compute_prediction(
        filepaths=filepaths,
        model_version=model_version,
        organelle=organelle,
        trg_scale=trg_scale,
        force_prediction=force_prediction,
        models_folder=models_folder,
        use_ensemble=use_ensemble,
    )


if __name__ == "__main__":
    app()
