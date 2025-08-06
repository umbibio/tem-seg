from pathlib import Path

__all__ = [
    "MODULE_PATH",
    "CONFIG_FILES",
]

MODULE_PATH = Path(__file__).parent.parent

# Files to load in order (later files override earlier ones)
CONFIG_FILES = [
    MODULE_PATH / "settings.toml",
    Path.cwd() / "settings.toml",
]
