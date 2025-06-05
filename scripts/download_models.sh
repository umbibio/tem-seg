#!/bin/bash

set -e

ZENODO_RECORD_ID="15602446"

BASE_URL="https://zenodo.org/records/${ZENODO_RECORD_ID}/files/"

files=(
    "tem-seg-models_v1.0.0.tar.gz"
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
