from pathlib import Path
import re

from tem_analysis_pipeline import prediction_tools
from tem_analysis_pipeline.calibration import (
    NoScaleError,
    NoScaleNumberError,
    NoScaleUnitError,
)


if __name__ == "__main__":
    import argparse

    class ns(argparse.Namespace):
        filepaths: list[Path]
        model_version: str
        organelle: str
        trg_scale: float
        force_prediction: bool
        models_folder: str | None
        use_ensemble: bool

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", type=Path, nargs="+")
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--organelle", type=str, required=True)
    parser.add_argument("--trg-scale", type=float, required=True)
    parser.add_argument("--force-prediction", action="store_true")
    parser.add_argument("--models-folder", type=Path, default=None)
    parser.add_argument("--use-ensemble", action="store_true")
    args: ns = parser.parse_args()

    predictions_basedir = Path("prediction").joinpath(
        args.model_version, args.organelle
    )

    prediction_tools.select_model_version(
        args.model_version,
        models_folder=args.models_folder,
        use_ensemble=args.use_ensemble,
    )
    model = None

    for img_filepath in args.filepaths:
        assert img_filepath.exists(), f"Image file {img_filepath} does not exist"

        output_basedir = img_filepath.parent.joinpath(predictions_basedir)

        prd_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}.png", img_filepath.name)
        )
        sft_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}-soft.png", img_filepath.name)
        )
        print(img_filepath, flush=True)

        img = prediction_tools.load_image(img_filepath.as_posix())

        if not prd_filepath.exists() or args.force_prediction:
            prd_filepath.parent.mkdir(parents=True, exist_ok=True)

            if model is None:
                model = prediction_tools.get_organelle_ensemble_model(args.organelle)

            try:
                prd = prediction_tools.image_prediction(
                    img, model, trg_scale=args.trg_scale
                )
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
