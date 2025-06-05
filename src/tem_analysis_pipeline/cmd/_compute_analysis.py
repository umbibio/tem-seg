import json
import re
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from PIL import Image

from .. import calibration, prediction_tools
from ..analysis import _analysis as analysis
from ..calibration import NoScaleError, NoScaleNumberError, NoScaleUnitError
from ..model.config import config


def compute_analysis(
    study_name: str,
    model_name: str,
    organelle: str,
    redo_analysis: bool = False,
    force_convolution: bool = False,
    n_jobs: int = 1,
) -> None:
    """
    Compute analysis for the given image files using the specified model predictions.

    Args:
        filepaths: List of paths to the image files to process
        model_version: Version of the model to use
        organelle: Target organelle for the analysis
        trg_scale: Target scale for the analysis in um/px
        redo_analysis: Whether to redo analysis even if output file exists
        force_convolution: Whether to force convolution even if soft prediction exists
        models_folder: Optional folder containing the models
        use_ensemble: Whether to use ensemble model
        cross_validation_kfolds: Number of folds for cross-validation
        n_jobs: Number of parallel jobs to run
    """
    studies_basedir = Path("studies")
    study_dir = studies_basedir / study_name
    target_scale = config[organelle]["target_scale"]

    def worker_task(img_filepath):
        img_filepath = Path(img_filepath)
        img_name = img_filepath.name

        assert img_filepath.exists(), f"Image file {img_filepath} does not exist"

        p = re.compile(".tif$")
        prd_filepath = predictions_dir / p.sub(f"-{organelle}.png", img_name)
        sft_filepath = predictions_dir / p.sub(f"-{organelle}-soft.png", img_name)
        jsn_filepath = predictions_dir / p.sub(f"-{organelle}.json", img_name)
        csv_filepath = predictions_dir / p.sub(f"-{organelle}.csv", img_name)
        geo_filepath = predictions_dir / p.sub(f"-{organelle}.geojson", img_name)

        if jsn_filepath.exists() and not redo_analysis:
            print("Already analyzed:", jsn_filepath, flush=True)
            return

        if not prd_filepath.exists():
            print("Not available:", prd_filepath, flush=True)
            return

        print(prd_filepath, flush=True)
        try:
            img = prediction_tools.load_image(img_filepath.as_posix())

            try:
                img_scale = calibration.get_calibration(img)
            except (NoScaleError, NoScaleNumberError, NoScaleUnitError) as e:
                print(
                    f"Warning: could not get scale for {img_filepath}: {e}",
                    flush=True,
                )
                return

            if not sft_filepath.exists() or force_convolution:
                print("Computing soft prediction ...", flush=True)
                prd = Image.open(prd_filepath)
                sft = prediction_tools.convolve_prediction(
                    prd, img_scale, img_scale, target_scale * 2 / 3, 24
                )
                sft_filepath.parent.mkdir(parents=True, exist_ok=True)
                sft.save(sft_filepath)
            else:
                sft = Image.open(sft_filepath)

            prd = prediction_tools.threshold_prediction(sft, threshold=0.5)
            prd = prediction_tools.remove_small_predictions(
                prd, img_scale, min_area_um2=0.018
            )

            print("Analyzing prediction ...", flush=True)
            data, annotations = analysis.analyze_organelle_prediction(
                img, prd, img_scale, organelle
            )

            jsn_filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(jsn_filepath, "w") as file:
                json.dump(data[organelle], file)

            csv_filepath.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(data[organelle])
            df.to_csv(csv_filepath)

            geo_filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(geo_filepath, "w") as file:
                json.dump(annotations, file)

        except Exception as e:
            print(f"Error processing {img_filepath}: {e}", flush=True)
            return

    for condition_dir in study_dir.iterdir():
        if not condition_dir.is_dir():
            continue

        predictions_dir = condition_dir / "prediction" / model_name / organelle

        image_list = sorted(condition_dir.glob("*.tif"))

        Parallel(n_jobs=n_jobs)(
            delayed(worker_task)(img_filepath) for img_filepath in image_list
        )
        print("\n")
