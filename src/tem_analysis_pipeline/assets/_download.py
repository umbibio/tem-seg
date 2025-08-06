from pathlib import Path

from ..config import settings
from ..utils import download_and_extract


def download_model_weights(assets_folder: str | Path | None = None) -> None:
    if assets_folder is None:
        assets_folder = Path(settings.assets.path)
    else:
        assets_folder = Path(assets_folder)
    assets_folder.mkdir(parents=True, exist_ok=True)

    model_weights_cfg = settings.assets.model_weights
    download_and_extract(
        urls=model_weights_cfg.mirrors,
        archive_filename=assets_folder / model_weights_cfg.filename,
        extract_to=assets_folder,
        expected_hash=model_weights_cfg.sha256,
        expected_hash_algorithm="sha256",
        cleanup_archive=False,
    )


def get_models_folder(architecture: str, download: bool = True) -> Path:
    assets_folder = Path(settings.assets.path)

    models_folder = assets_folder / "models" / architecture
    if models_folder.exists():
        return models_folder

    if (
        architecture == "unet"
        and (assets_folder / "models" / "5-fold_cross_validation").exists()
    ):
        models_folder = assets_folder / "models"
        return models_folder

    if not download:
        raise FileNotFoundError(f"Models folder not found at {models_folder}")

    download_model_weights(assets_folder)
    return get_models_folder(architecture, download=False)
