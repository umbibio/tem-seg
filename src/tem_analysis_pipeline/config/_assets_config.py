import re
from typing import List

from pydantic import BaseModel, Field, field_validator

from ._config import MODULE_PATH


class ModelWeightsConfig(BaseModel):
    """Configuration for model weights."""

    mirrors: List[str] = Field(..., description="List of mirror URLs for downloading")
    filename: str = Field(..., description="Filename of the model weights")
    md5: str = Field(..., min_length=32, max_length=32, description="MD5 hash")
    sha256: str = Field(..., min_length=64, max_length=64, description="SHA256 hash")

    @field_validator("md5")
    @classmethod
    def validate_md5(cls, v: str) -> str:
        if not re.match(r"^[a-fA-F0-9]{32}$", v):
            raise ValueError("MD5 must be a 32-character hexadecimal string")
        return v.lower()

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, v: str) -> str:
        if not re.match(r"^[a-fA-F0-9]{64}$", v):
            raise ValueError("SHA256 must be a 64-character hexadecimal string")
        return v.lower()


class AssetsConfig(BaseModel):
    """Configuration for assets."""

    path: str = Field(
        default=(MODULE_PATH / "assets").as_posix(),
        description="Path to assets directory",
    )
    model_weights: ModelWeightsConfig
