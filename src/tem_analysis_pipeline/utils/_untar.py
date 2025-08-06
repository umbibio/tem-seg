import tarfile
from pathlib import Path

from rich.console import Console
from rich.progress import Progress

console = Console()


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract tar.gz with progress bar and safety checks.

    Returns:
        True if extraction successful, False otherwise
    """
    try:
        # Create extraction directory
        extract_to.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()

            with Progress(console=console) as progress:
                task = progress.add_task(
                    f"📦 Extracting {archive_path.name}...", total=len(members)
                )

                for member in members:
                    # Safety check
                    if not _is_safe_path(member.name, extract_to):
                        console.print(
                            f"⚠️  Skipping unsafe path: {member.name}", style="yellow"
                        )
                        continue

                    # Skip special files
                    if not (member.isfile() or member.isdir()):
                        console.print(
                            f"⚠️  Skipping special file: {member.name}", style="yellow"
                        )
                        continue

                    tar.extract(member, extract_to)
                    progress.update(task, advance=1)

        console.print(f"✅ Successfully extracted to {extract_to}", style="green")
        return True

    except Exception as e:
        console.print(f"❌ Extraction failed: {e}", style="red")
        return False


def _is_safe_path(path: str, extract_to: Path) -> bool:
    """Check if extraction path is safe (no directory traversal)."""
    try:
        member_path = (extract_to / path).resolve()
        return member_path.is_relative_to(extract_to.resolve())
    except (ValueError, OSError):
        return False
