from __future__ import annotations

from pathlib import Path
import csv

from tem_seg import prediction_tools
from tem_seg.calibration import (
    NoScaleError,
    NoScaleNumberError,
    NoScaleUnitError,
    get_calibration,
)


def init_pixel_sizes_tsv(filepaths: list[Path], output: Path) -> None:
    """Create a TSV file with the calibrated pixel size of the input images.

    Each row contains the image filename and the calibrated pixel size in
    nanometers per pixel. If the calibration cannot be parsed, the value is
    left empty.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["filename", "pixel_size_nm"])
        for path in filepaths:
            pixel_size: str | float = ""
            if path.exists():
                img = prediction_tools.load_image(path.as_posix())
                try:
                    pixel_size = get_calibration(img) * 1000
                except (NoScaleError, NoScaleNumberError, NoScaleUnitError):
                    pixel_size = ""
            writer.writerow([path.name, pixel_size])
