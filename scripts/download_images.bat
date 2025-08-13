REM run as
REM cmd /c download_images.bat

@echo off
setlocal EnableExtensions

REM Config
set "ZENODO_RECORD_ID=15602048"
set "BASE_URL=https://zenodo.org/records/%ZENODO_RECORD_ID%/files/"

REM Files to fetch
for %%F in (
  tem-seg-data_slide_images.tar.gz
  tem-seg-data_mitochondria_masks.tar.gz
) do (
    if not exist "%%F" (
        echo(
        echo Downloading %%F
        curl.exe -L -o "%%F" "%BASE_URL%%%F?download=1" || exit /b 1

        echo Download complete. Extracting...
        tar -xzf "%%F" || exit /b 1

        echo Extraction complete.
    )
)

endlocal
