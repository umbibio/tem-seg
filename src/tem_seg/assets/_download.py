import re
from pathlib import Path

from ..config import settings
from ..utils import download_and_extract


def download_model_weights(
    architecture: str = "unet",
    name: str = "Mixture",
    kind: str = "single_fold",
    assets_folder: str | Path | None = None,
) -> None:
    if assets_folder is None:
        assets_folder = Path(settings.assets.path).expanduser()
    else:
        assets_folder = Path(assets_folder)
    assets_folder.mkdir(parents=True, exist_ok=True)

    if kind == "single_fold":
        k = "k1"
    elif (m := re.match(r"(\d)-fold_cross_validation", kind)) is not None:
        k = f"k{m.group(1)}"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    model_weights_cfg = settings.assets.model_weights[architecture][name][k]
    download_and_extract(
        urls=model_weights_cfg.mirrors,
        archive_filename=assets_folder / model_weights_cfg.filename,
        extract_to=assets_folder,
        expected_hash=model_weights_cfg.sha256,
        expected_hash_algorithm="sha256",
        cleanup_archive=False,
    )


def get_model_dir(
    architecture: str,
    name: str,
    kind: str,
    download: bool = True,
) -> Path:
    assets_folder = Path(settings.assets.path).expanduser()

    model_dir = assets_folder / "models" / architecture / name / kind
    if model_dir.exists():
        return model_dir

    if not download:
        raise FileNotFoundError(f"Models folder not found at {model_dir}")

    download_model_weights(architecture, name, kind)
    return get_model_dir(architecture, name, kind, download=False)
