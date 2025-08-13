# run as
# powershell -ExecutionPolicy Bypass -File make_tfrecords.ps1

# Stop on first error (like `set -e`)
$ErrorActionPreference = "Stop"

# Environment variable
$env:CUDA_VISIBLE_DEVICES = ""

# Datasets
$DATASET_NAMES = @(
    'DRP1-KO',
    'HCI-010',
    'Mixture',
    'PIM001-P'
)

foreach ($name in $DATASET_NAMES) {
    Write-Host "Processing $name"

    # Training and validation
    tem-seg preprocess tfrecords `
        $name `
        "data/$name/tra_val/slide_images/" `
        "data/$name/tra_val/mitochondria/masks/" `
        -o mitochondria `
        --test-size 0  # 0 = no test split here

    # Test (skip for Mixture)
    if ($name -eq 'Mixture') {
        # Mixture dataset does not have a test set
        continue
    }

    tem-seg preprocess tfrecords `
        $name `
        "data/$name/tst/slide_images/" `
        "data/$name/tst/mitochondria/masks/" `
        -o mitochondria `
        --test-size -1  # -1 = use all images for testing
}
