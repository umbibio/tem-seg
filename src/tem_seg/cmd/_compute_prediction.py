from pathlib import Path
from typing import Literal

from tem_seg import prediction_tools
from tem_seg.calibration import (
    NoScaleError,
    NoScaleNumberError,
    NoScaleUnitError,
    get_calibration,
)


def compute_prediction(
    filepaths: list[Path],
    model_architecture: Literal["unet", "unetpp"],
    model_name: str,
    organelle: str,
    force_prediction: bool = False,
    models_folder: str | Path | None = None,
    use_ensemble: bool = False,
    checkpoint: Literal["last", "best_loss"] = "last",
    cross_validation_kfolds: int | None = None,
    round_output: bool = False,
    pixel_size_nm: float | None = None,
) -> None:
    """
    Compute predictions for the given image files using the specified model.

    Args:
        filepaths: List of paths to the image files to process
        model_architecture: Architecture of the model
        model_name: Name of the model to use
        organelle: Target organelle for the prediction
        force_prediction: Whether to force prediction even if output file exists
        models_folder: Optional folder containing the models
        use_ensemble: Whether to use ensemble model
        checkpoint: Checkpoint to use
        cross_validation_kfolds: Number of cross-validation folds
        round_output: Whether to round the predictions
        pixel_size_nm: Calibrated pixel size in nm/pixel. If provided, bypass automatic
            calibration detection.
    """
    match model_architecture:
        case "unet":
            print("Using U-Net architecture")
            from ..model.unet import unet_config as config

        case "unetpp":
            print("Using U-Net++ architecture")
            from ..model.unetpp import unetpp_config as config

        case _:
            raise ValueError(f"Unknown model architecture: {model_architecture}")

    prediction_tools.select_model_version(
        model_architecture=model_architecture,
        model_name=model_name,
        models_folder=models_folder,
        use_ensemble=use_ensemble,
        cross_validation_kfolds=cross_validation_kfolds,
    )

    trg_scale = config[organelle]["target_scale"]

    img_scales: dict[str, float] = {}
    if pixel_size_nm is not None:
        for img_filepath in filepaths:
            img_scales[img_filepath.name] = pixel_size_nm / 1000
    else:
        for img_filepath in filepaths:
            assert img_filepath.exists(), f"Image file {img_filepath} does not exist"
            img = prediction_tools.load_image(img_filepath.as_posix())
            try:
                img_scales[img_filepath.name] = get_calibration(img)
                print(
                    f"Image: {img_filepath.name}, Parsed Calibrated Pixel Size: {img_scales[img_filepath.name]} nm/pixel",
                    flush=True,
                )
            except NoScaleError as e:
                print(
                    f"Warning: no scale could be read for image {img_filepath}",
                    flush=True,
                )
                print(e, flush=True)
            except NoScaleNumberError as e:
                print(
                    f"Warning: no scale number could be read for image {img_filepath}",
                    flush=True,
                )
                print(e, flush=True)
            except NoScaleUnitError as e:
                print(
                    f"Warning: no scale unit could be read for image {img_filepath}",
                    flush=True,
                )
                print(e, flush=True)

        filepaths = [p for p in filepaths if p.name in img_scales]

    model_id = f"{model_architecture}_{model_name}"
    if use_ensemble:
        model_id += f"_k{cross_validation_kfolds}"

    predictions_basedir = Path("prediction") / model_id / organelle

    model = None

    for img_filepath in filepaths:
        assert img_filepath.exists(), f"Image file {img_filepath} does not exist"

        output_basedir = img_filepath.parent / predictions_basedir

        prd_filepath = output_basedir / (img_filepath.stem + f"-{organelle}.png")
        print(img_filepath, flush=True)

        img = prediction_tools.load_image(img_filepath.as_posix())
        img_scale = img_scales.get(img_filepath.name)

        if img_scale is None:
            print(
                f"Warning: skipping image {img_filepath} due to missing calibration",
                flush=True,
            )
            continue

        if not prd_filepath.exists() or force_prediction:
            prd_filepath.parent.mkdir(parents=True, exist_ok=True)

            if model is None:
                model = prediction_tools.get_organelle_ensemble_model(
                    organelle, ckpt=checkpoint, round_output=round_output
                )

            try:
                prd = prediction_tools.image_prediction(
                    img, model, trg_scale=trg_scale, img_scale=img_scale
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
