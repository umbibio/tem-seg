from pydantic_settings import BaseSettings, SettingsConfigDict

from ._assets_config import AssetsConfig
from ._config import CONFIG_FILES
from ._environment_config import EnvironmentConfig
from ._utils import load_config_files

__all__ = [
    "Settings",
    "settings",
]


class Settings(BaseSettings):
    """Main application settings."""

    assets: AssetsConfig
    environment: EnvironmentConfig

    model_config = SettingsConfigDict(
        # Environment variable settings
        env_prefix="TEM_SEG_",
        env_nested_delimiter="__",
        case_sensitive=False,
        validate_assignment=True,
        extra="ignore",
    )


# Load configuration and create settings
try:
    _config_data = load_config_files(CONFIG_FILES)
    if not _config_data:
        print(
            "⚠️  No configuration files found, using defaults/environment variables only"
        )

    settings = Settings(**_config_data)

except Exception as e:
    print(f"❌ Error creating settings: {e}")
    raise
