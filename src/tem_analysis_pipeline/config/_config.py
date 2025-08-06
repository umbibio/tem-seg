from pathlib import Path

__all__ = [
    "CONFIG_FILES",
    "DEFAULT_SETTINGS_PATH",
    "MODULE_PATH",
    "TEM_SEG_LOCAL_PATH",
]

MODULE_PATH = Path(__file__).parent.parent

TEM_SEG_LOCAL_PATH = Path("~/.tem-seg").expanduser()

DEFAULT_SETTINGS_PATH = MODULE_PATH / "settings.toml"

# Files to load in order (later files override earlier ones)
CONFIG_FILES = [
    DEFAULT_SETTINGS_PATH,
    Path.cwd() / "settings.toml",
]
