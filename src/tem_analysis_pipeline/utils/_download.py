import hashlib
from pathlib import Path
from ssl import create_default_context
from typing import List
from urllib.error import URLError
from urllib.request import Request, urlopen

import typer
from certifi import where
from rich.console import Console
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from ._untar import extract_archive

console = Console()


def calculate_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Calculate hash of a file."""
    hash_func = hashlib.new(algorithm)
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def verify_existing_file(path: Path, expected_hash: str, hash_algorithm: str) -> bool:
    """
    Check if file exists and has correct hash.

    Returns:
        True if file exists and hash matches, False otherwise
    """
    if not path.exists():
        return False

    console.print(
        f"📁 File {path.name} already exists, checking integrity...", style="blue"
    )

    try:
        existing_hash = calculate_file_hash(path, hash_algorithm)
        if existing_hash == expected_hash:
            console.print(
                "✅ File integrity verified, skipping download", style="green"
            )
            return True
        else:
            console.print(
                "⚠️  File exists but hash mismatch, will re-download", style="yellow"
            )
            path.unlink()  # Remove corrupted file
            return False
    except Exception as e:
        console.print(f"⚠️  Error checking existing file: {e}", style="yellow")
        path.unlink()  # Remove problematic file
        return False


def create_request(url: str) -> Request:
    """Create HTTP request with appropriate headers."""
    return Request(url, headers={"User-agent": "tem-seg-user"})


def open_url_with_fallback(req: Request) -> object:
    """
    Open URL with certificate fallback.

    Returns:
        Opened URL response object

    Raises:
        Exception if both attempts fail
    """
    try:
        return urlopen(req)

    except URLError:
        console.print(
            "⚠️  Failed with default certificates, trying with certifi...",
            style="yellow",
        )
        return urlopen(req, context=create_default_context(cafile=where()))


def download_single_file(url: str, temp_path: Path) -> bool:
    """
    Download file from a single URL to temporary path.

    Returns:
        True if download successful, False otherwise
    """
    blocksize = 1024 * 8

    try:
        req = create_request(url)
        open_url = open_url_with_fallback(req)

        with open_url as resp:
            total = resp.info().get("content-length", None)
            total_size = int(total) if total else None

            with Progress(
                TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "download",
                    filename=temp_path.stem,  # Remove .tmp extension for display
                    total=total_size,
                )

                with temp_path.open("wb") as f:
                    while True:
                        block = resp.read(blocksize)
                        if not block:
                            break
                        f.write(block)
                        progress.update(task, advance=len(block))

        return True

    except Exception as e:
        console.print(f"❌ Download failed: {e}", style="red")
        return False


def verify_downloaded_file(
    temp_path: Path, expected_hash: str, hash_algorithm: str
) -> bool:
    """
    Verify integrity of downloaded file.

    Returns:
        True if hash matches, False otherwise
    """
    console.print("🔍 Verifying file integrity...", style="blue")

    try:
        actual_hash = calculate_file_hash(temp_path, hash_algorithm)

        if actual_hash == expected_hash:
            console.print("✅ Download successful and verified!", style="green")
            return True

        else:
            console.print(
                f"❌ Hash mismatch! Expected: {expected_hash[:16]}..., Got: {actual_hash[:16]}...",
                style="red",
            )
            return False

    except Exception as e:
        console.print(f"❌ Error verifying file: {e}", style="red")
        return False


def cleanup_temp_file(temp_path: Path) -> None:
    """Remove temporary file if it exists."""
    if temp_path.exists():
        temp_path.unlink()


def try_download_from_url(
    url: str, path: Path, expected_hash: str, hash_algorithm: str
) -> bool:
    """
    Attempt to download and verify file from a single URL.

    Returns:
        True if successful, False otherwise
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        # Download the file
        if not download_single_file(url, temp_path):
            cleanup_temp_file(temp_path)
            return False

        # Verify the hash
        if not verify_downloaded_file(temp_path, expected_hash, hash_algorithm):
            cleanup_temp_file(temp_path)
            return False

        # Move temp file to final location
        temp_path.rename(path)
        return True

    except KeyboardInterrupt:
        console.print("❌ Download cancelled by user", style="red")
        cleanup_temp_file(temp_path)
        raise typer.Abort()

    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        cleanup_temp_file(temp_path)
        return False


def download_with_fallback(
    urls: List[str],
    path: Path,
    expected_hash: str,
    hash_algorithm: str = "sha256",
) -> bool:
    """
    Download file from a list of URLs with hash verification and fallback.

    Args:
        urls: List of URLs to try in order
        path: Where to save the file
        expected_hash: Expected hash of the file
        hash_algorithm: Hash algorithm to use (default: sha256)

    Returns:
        True if download and verification successful, False otherwise
    """
    # Check if file already exists and is valid
    if verify_existing_file(path, expected_hash, hash_algorithm):
        return True

    # Try each URL in order
    for i, url in enumerate(urls):
        console.print(f"🌐 Trying URL {i + 1}/{len(urls)}: {url}", style="blue")

        if try_download_from_url(url, path, expected_hash, hash_algorithm):
            return True

        if i < len(urls) - 1:
            console.print("🔄 Trying next URL...", style="yellow")

    console.print("❌ All download attempts failed", style="red")
    return False


def download_file(
    mirrors: List[str] | str, filepath: str | Path, expected_sha256: str
) -> bool:
    """For a single file with multiple mirrors."""
    if isinstance(mirrors, str):
        mirrors = [mirrors]

    return download_with_fallback(
        urls=mirrors,
        path=Path(filepath),
        expected_hash=expected_sha256,
    )


def download_and_extract(
    urls: List[str],
    archive_filename: str,
    extract_to: Path,
    expected_hash: str,
    expected_hash_algorithm: str = "sha256",
    cleanup_archive: bool = True,
) -> bool:
    """
    Download tar.gz file and extract it.

    Args:
        urls: URLs to try downloading from
        archive_filename: Local filename for the archive
        expected_hash: Expected SHA256 hash
        extract_to: Directory to extract files to
        cleanup_archive: Whether to delete archive after extraction

    Returns:
        True if successful, False otherwise
    """
    archive_path = Path(archive_filename)

    # Download the archive
    if not download_with_fallback(
        urls, archive_path, expected_hash, expected_hash_algorithm
    ):
        return False

    # Extract the archive
    success = extract_archive(archive_path, extract_to)

    # Cleanup if requested and extraction was successful
    if cleanup_archive and success and archive_path.exists():
        console.print(f"🗑️  Removing archive {archive_path.name}", style="blue")
        archive_path.unlink()

    return success
