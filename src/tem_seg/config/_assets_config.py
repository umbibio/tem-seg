from typing import Dict

from pydantic import BaseModel, Field

from ._config import TEM_SEG_LOCAL_PATH
from ._downloads_config import DownloadableFileConfig


class AssetsConfig(BaseModel):
    """Configuration for assets."""

    path: str = Field(
        default=(TEM_SEG_LOCAL_PATH / "assets").as_posix(),
        description="Path to assets directory",
    )
    model_weights: Dict[
        str,  # architecture
        Dict[
            str,  # name
            Dict[
                str,  # kind (single_fold, 5-fold_cross_validation)
                DownloadableFileConfig,
            ],
        ],
    ]
