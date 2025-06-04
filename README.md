# TEM Analysis Pipeline

## Overview

This repository contains a pipeline for the analysis of mitochondria morphology using semantic segmentation on Transmission Electron Microscopy (TEM) images. The pipeline includes tools for preprocessing TEM images, training U-Net models for semantic segmentation, and generating predictions on new images.

## Status

**Note:** Previous codebase was developed for an old version of TensorFlow (2.9) and has been updated to be compatible with TensorFlow 2.19. The training functionality is still being tested and may not work as expected.

## Installation

### Requirements

- Python 3.12
- TensorFlow 2.19 (with CUDA support recommended for training)
- Other dependencies as listed in pyproject.toml

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/yourusername/tem-paper-reproducibility.git
cd tem-paper-reproducibility

# Install the package in development mode
pip install -e .
```

## Usage

The package provides a command-line interface with three main commands: `preprocess`, `train`, and `predict`.

### Preprocessing Data

Before training, you need to preprocess your TEM images and masks into TFRecord format:

```bash
tem-seg-model preprocess tfrecords \
    /path/to/slides \
    /path/to/masks \
    --organelle mitochondria \
    --output-dirpath /path/to/output \
    --slide-format tif
```

### Training a Model

To train a U-Net model for semantic segmentation:

```bash
tem-seg-model train \
    /path/to/working/directory \
    --organelle mitochondria \
    --k-fold 1 \
    --n-epochs-per-run 1200
```

### Making Predictions

To generate predictions using a trained model:

```bash
tem-seg-model predict \
    /path/to/images/*.tif \
    --model-version Mixture \
    --organelle mitochondria \
    --trg-scale 0.0075 \
    --use-ensemble
```

## Project Structure

```
src/tem_analysis_pipeline/
├── cmd/                    # Command-line interface modules
│   ├── __init__.py         # Main CLI entry point
│   ├── _compute_prediction.py  # Prediction functionality
│   ├── _preprocess.py      # Data preprocessing
│   └── _train.py           # Model training
├── model/                  # Model architecture and utilities
│   ├── __init__.py
│   ├── _unet.py            # U-Net model implementation
│   ├── config.py           # Configuration parameters
│   ├── custom_objects.py   # Custom Keras objects
│   ├── losses.py           # Loss functions
│   ├── metrics.py          # Evaluation metrics
│   └── utils.py            # Utility functions
├── calibration.py          # Image calibration utilities
└── prediction_tools.py     # Prediction utilities
```

## Data Requirements

### Input Images

- TEM images should be in TIFF format (or other formats supported by PIL)
- Masks should be single-channel images (mode 'L' or 'P' in PIL)
- Images should be properly calibrated with scale information displayed on the image. This should work for the images used in this study, but may not work for other images if the scale format is different.

### Dataset Organization

TODO

## Configuration

Model and training parameters are configured in `src/tem_analysis_pipeline/model/config.py`. Key parameters include:

- `tile_shape`: Input image size for the model
- `window_shape`: Output mask size for the model
- `target_scale`: Target scale for image preprocessing
- `fraction_of_empty_to_keep`: Fraction of empty masks to keep during training

These parameters can be customized for different organelles (cell, mitochondria, nucleus).

## License

TODO

## Citation

TODO

## Acknowledgments

TODO
