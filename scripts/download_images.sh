#!/bin/bash

set -e

BASE_URL="https://zenodo.org/records/15602048/files/"

files=(
    "tem-seg-data_slide_images.tar.gz"
    "tem-seg-data_mitochondria_masks.tar.gz"
)

for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo
        echo "Downloading $file"
        wget -O "$file" "${BASE_URL}${file}?download=1"
        echo "Download complete. Extracting..."
        tar -xzf "$file"
        echo "Extraction complete."
    fi
done
