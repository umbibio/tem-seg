REM run as
REM cmd /c make_tfrecords.bat

@echo off
setlocal EnableExtensions

REM Stop on first error: make each critical command fail the script
REM (use `|| exit /b 1` after each tem-seg call)

REM Environment variable
set "CUDA_VISIBLE_DEVICES="

REM Datasets
for %%A in (DRP1-KO HCI-010 Mixture PIM001-P) do (
    echo Processing %%A

    REM Training and validation
    tem-seg preprocess tfrecords ^
        %%A ^
        "data/%%A/tra_val/slide_images/" ^
        "data/%%A/tra_val/mitochondria/masks/" ^
        -o mitochondria ^
        --test-size 0 || exit /b 1

    REM Test (skip for Mixture)
    if /I "%%A"=="Mixture" (
        REM Mixture dataset does not have a test set
    ) else (
        tem-seg preprocess tfrecords ^
            %%A ^
            "data/%%A/tst/slide_images/" ^
            "data/%%A/tst/mitochondria/masks/" ^
            -o mitochondria ^
            --test-size -1 || exit /b 1
    )
)

endlocal
