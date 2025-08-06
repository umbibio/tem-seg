from pathlib import Path

__all__ = [
    "MODULE_PATH",
    "DEFAULT_SETTINGS_PATH",
    "CONFIG_FILES",
]

MODULE_PATH = Path(__file__).parent.parent

DEFAULT_SETTINGS_PATH = MODULE_PATH / "settings.toml"

# Files to load in order (later files override earlier ones)
CONFIG_FILES = [
    DEFAULT_SETTINGS_PATH,
    Path.cwd() / "settings.toml",
]
