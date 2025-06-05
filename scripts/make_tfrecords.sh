#!/bin/bash

set -e

DATASET_NAMES=(
    DRP1-KO
    HCI-010
    PIM001-P
)

export CUDA_VISIBLE_DEVICES=""
DATASET_NAMES=(
    DRP1-KO
    HCI-010
    Mixture
    PIM001-P
)

for name in "${DATASET_NAMES[@]}"; do
    echo "Processing $name"

    # Training and validation
    tem-seg preprocess tfrecords \
        $name \
        data/$name/tra_val/slide_images/ \
        data/$name/tra_val/mitochondria/masks/ \
        -o mitochondria \
        --test-size 0 # Test size of 0 means no test set

    # Test
    if [ "$name" == "Mixture" ]; then
        # Mixture dataset does not have a test set
        continue
    fi
    tem-seg preprocess tfrecords \
        $name \
        data/$name/tst/slide_images/ \
        data/$name/tst/mitochondria/masks/ \
        -o mitochondria \
        --test-size -1 # Test size of -1 means all images are used for testing

done
