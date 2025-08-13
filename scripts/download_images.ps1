# run as
# powershell -ExecutionPolicy Bypass -File download_images.ps1

# Stop on first error
$ErrorActionPreference = "Stop"

# Config
$ZENODO_RECORD_ID = "15602048"
$BASE_URL = "https://zenodo.org/records/$ZENODO_RECORD_ID/files/"

# Files to fetch
$files = @(
  "tem-seg-data_slide_images.tar.gz",
  "tem-seg-data_mitochondria_masks.tar.gz"
)

foreach ($file in $files) {
    if (-not (Test-Path -Path $file -PathType Leaf)) {
        Write-Host ""
        Write-Host "Downloading $file"
        $url = "$BASE_URL${file}?download=1"
        Write-Host "Downloading from $url"
        Invoke-WebRequest -Uri $url -OutFile $file
        Write-Host "Download complete. Extracting..."
        tar -xzf $file
        Write-Host "Extraction complete."
    }
}
