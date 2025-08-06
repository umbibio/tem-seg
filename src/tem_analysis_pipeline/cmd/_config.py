from pathlib import Path

from rich.console import Console

console = Console()


def init_config():
    import shutil

    from ..config import DEFAULT_SETTINGS_PATH

    target_path = Path.cwd() / DEFAULT_SETTINGS_PATH.name

    shutil.copyfile(
        DEFAULT_SETTINGS_PATH,
        target_path,
    )

    console.print(
        f"✅ Copied default configuration file to {target_path}", style="green"
    )
