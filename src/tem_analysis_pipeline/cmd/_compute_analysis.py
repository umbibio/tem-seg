import json
import re
from pathlib import Path


from joblib import Parallel, delayed
from PIL import Image

from .. import prediction_tools, calibration
from ..analysis import _analysis as analysis

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filepaths", type=str, nargs="+")
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--organelle", type=str, required=True)
    parser.add_argument("--trg-scale", type=float, required=True)
    parser.add_argument("--redo-analysis", action="store_true")
    parser.add_argument("--force-convolution", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    predictions_basedir = Path("prediction").joinpath(
        args.model_version, args.organelle
    )
    metadata_basedir = Path("metadata")
    # model_version is like vNN_string
    version_number = int(re.match("^v(\d+)", args.model_version).group(1))

    prediction_tools.select_model_version(args.model_version)

    def worker_task(img_filepath):
        img_filepath = Path(img_filepath)

        assert img_filepath.exists(), f"Image file {img_filepath} does not exist"

        output_basedir = img_filepath.parent.joinpath(predictions_basedir)

        prd_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}.png", img_filepath.name)
        )
        sft_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}-soft.png", img_filepath.name)
        )
        jsn_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}.json", img_filepath.name)
        )
        scs_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}-scores.json", img_filepath.name)
        )
        geo_filepath = output_basedir.joinpath(
            re.sub(".tif$", f"-{args.organelle}.geojson", img_filepath.name)
        )
        met_filepath = metadata_basedir.joinpath(
            re.sub(".tif$", ".json", img_filepath.name)
        )

        if jsn_filepath.exists() and not args.redo_analysis:
            print("Already analyzed:", jsn_filepath, flush=True)
            return

        if not prd_filepath.exists():
            print("Not available:", prd_filepath, flush=True)
            return

        print(prd_filepath, flush=True)
        img = prediction_tools.load_image(img_filepath.as_posix())

        if met_filepath.exists():
            with open(met_filepath, "r") as file:
                metadata = json.load(file)
                img_scale = metadata["scale_um_per_px"]
        else:
            img_scale = calibration.get_calibration(img)
            metadata = {"scale_um_per_px": img_scale}
            with open(met_filepath, "w") as file:
                json.dump(metadata, file)

        if not sft_filepath.exists() or args.force_convolution:
            print("Computing soft prediction ...", flush=True)
            prd = Image.open(prd_filepath)
            sft = prediction_tools.convolve_prediction(
                prd, img_scale, args.trg_scale, args.trg_scale * 2 / 3, 24
            )
            sft.save(sft_filepath)
        else:
            sft = Image.open(sft_filepath)

        prd = prediction_tools.threshold_prediction(sft, threshold=0.5)
        prd = prediction_tools.remove_small_predictions(
            prd, img_scale, min_area_um2=0.018
        )

        print("Analyzing prediction ...", flush=True)
        data, annotations = analysis.analyze_organelle_prediction(
            img, prd, img_scale, args.organelle
        )

        with open(jsn_filepath, "w") as file:
            json.dump(data[args.organelle], file)

        with open(geo_filepath, "w") as file:
            json.dump(annotations, file)

    Parallel(n_jobs=args.n_jobs)(
        delayed(worker_task)(img_filepath) for img_filepath in args.filepaths
    )
    print("\n")
