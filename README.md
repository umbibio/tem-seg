# TEM Analysis Pipeline

## Overview

This repository contains a pipeline for the analysis of mitochondria morphology using semantic segmentation on Transmission Electron Microscopy (TEM) images. The pipeline includes tools for preprocessing TEM images, training U-Net models for semantic segmentation, and generating predictions on new images.

## Status

The datasets are now available for download. See the [Data Download](#data-download) section for instructions.

**Note:** Previous codebase was developed for an old version of TensorFlow (2.9) and has been updated to be compatible with TensorFlow 2.19. We have tested the codebase and it works as expected. Please file an issue if you encounter any problems. Thank you for your patience.

## Installation

### Requirements

- Python 3.12
- TensorFlow 2.19 (with CUDA support recommended for training)
- Other dependencies as listed in pyproject.toml

### Installation Steps

```bash
# Clone the repository
git clone https://github.com/umbibio/tem-seg-reproducibility.git
cd tem-paper-reproducibility

# Install the package in development mode
pip install -e .
```

## Data Download

The TEM image dataset and masks are hosted on Zenodo. You can download them using the provided script:

```bash
# Navigate to the repository root
cd tem-paper-reproducibility

# Run the download script
bash scripts/download_images.sh
```

This script will download and extract the following files:
- `tem-seg-data_slide_images.tar.gz`: Contains all TEM slide images
- `tem-seg-data_mitochondria_masks.tar.gz`: Contains all mitochondria masks

## Preparing TFRecords

After downloading the data, you need to convert the images and masks to TFRecord format for training:

```bash
# Run the TFRecord creation script
bash scripts/make_tfrecords.sh
```

This script will process the following datasets:
- DRP1-KO
- HCI-010
- Mixture
- PIM001-P

For each dataset, it creates TFRecords for both training/validation and test sets (except for the Mixture dataset which doesn't have a test set).

## Usage

The package provides a command-line interface with three main commands: `preprocess`, `train`, and `predict`.

### Preprocessing Data

Before training, you need to preprocess your TEM images and masks into TFRecord format:

```bash
tem-seg preprocess tfrecords \
    dataset_name \
    /path/to/slides \
    /path/to/masks \
    --organelle mitochondria \
    --slide-format tif \
    --test-size 0.1 \
    --random-state 42
```

This command will:
1. Split your data into training/validation and test sets
2. Convert images and masks to TFRecord format
3. Save the TFRecords in the `data/dataset_name` directory

### Training a Model

To train a U-Net model for semantic segmentation:

```bash
tem-seg train \
    dataset_name \
    --organelle mitochondria \
    --fold-n 1 \
    --total-folds 5 \
    --shuffle-training \
    --batch-size 16 \
    --n-epochs-per-run 1200
```

The training process includes:
- K-fold cross-validation (when total-folds > 1)
- Automatic model checkpointing for best validation performance
- Evaluation metrics including F1 and F2 scores
- JSON logging of training history and evaluation results

### Making Predictions

To generate predictions using a trained model:

```bash
tem-seg predict \
    /path/to/images/*.tif \
    --model-version Mixture \
    --organelle mitochondria \
    --use-ensemble \
    --cross-validation-kfolds 5 \
    --checkpoint last
```

## Project Structure

```
src/tem_analysis_pipeline/
├── cmd/                    # Command-line interface modules
│   ├── __init__.py         # Main CLI entry point with Typer commands
│   ├── _compute_prediction.py  # Prediction functionality
│   ├── _preprocess.py      # Data preprocessing with train/test splitting
│   └── _train.py           # Model training with k-fold cross-validation
├── model/                  # Model architecture and utilities
│   ├── __init__.py
│   ├── _unet.py            # U-Net model implementation
│   ├── config.py           # Configuration parameters
│   ├── custom_objects.py   # Custom Keras objects
│   ├── losses.py           # Loss functions
│   ├── metrics.py          # Evaluation metrics with F1 and F2 scores
│   └── utils.py            # Utility functions
├── calibration.py          # Image calibration utilities
└── prediction_tools.py     # Prediction utilities
```

## Data Requirements

### Input Images

- TEM images should be in TIFF format (or other formats supported by PIL)
- Masks should be single-channel images (mode 'L' or 'P' in PIL)
- Images should be properly calibrated with scale information displayed on the image. This should work for the images used in this study, but may not work for other images if the scale format is different.
- Images and masks should have matching filenames for proper pairing during preprocessing.

### Dataset Organization

The pipeline expects the following dataset organization:

1. **Initial Data Structure**:
   ```
   /path/to/slides/  # Directory containing TEM images
   └── image1.tif
   └── image2.tif
   └── ...
   
   /path/to/masks/   # Directory containing mask images
   └── image1.png
   └── image2.png
   └── ...
   ```

2. **After Preprocessing**:
   ```
   data/
   └── dataset_name/
       ├── tra_val/organelle/tfrecords/
       │   └── *.tfrecord
       └── tst/organelle/tfrecords/
           └── *.tfrecord
   ```

3. **After Training**:
   
   For single-fold training:
   ```
   models/
   └── single_fold/dataset_name/
       └── mitochondria
           └── kf01
               ├── ckpt
               │   └── last.keras
               ├── evaluation
               │   ├── nnnnn_evaluation.json
               │   └── ...
               └── logs
                   ├── metrics.tsv
                   └── train
                       ├── events.out.tfevents.*.v2
   ```

   For k-fold cross-validation:
   ```
   models/
   └── 5-fold_cross_validation/dataset_name/
       └── mitochondria
           ├── kf01
           │   ├── ckpt
           │   │   ├── best_loss
           │   │   │   └── best_logs.json
           │   │   ├── best_loss.keras
           │   │   └── last.keras
           │   ├── evaluation
           │   │   ├── nnnnn_evaluation.json
           │   │   └── ...
           │   └── logs
           │       ├── metrics.tsv
           │       ├── train
           │       │   └── events.out.tfevents.*.v2
           │       └── validation
           │           └── events.out.tfevents.*.v2
           ├── kf02
           │   ├── ckpt
           │   │   ├── best_loss
           │   │   │   └── best_logs.json
           │   │   ├── best_loss.keras
           │   │   └── last.keras
           │   └── ...
           └── kf03
               └── ...
   ```

Each fold directory contains checkpoints, evaluation results, and TensorBoard logs for training and validation.

## Configuration

Model and training parameters are configured in `src/tem_analysis_pipeline/model/config.py`. Key parameters include:

- `tile_shape`: Input image size for the model
- `window_shape`: Output mask size for the model
- `target_scale`: Target scale for image preprocessing
- `fraction_of_empty_to_keep`: Fraction of empty masks to keep during training

These parameters can be customized for different organelles (cell, mitochondria, nucleus).

### Training Configuration

The training process can be customized with the following parameters:

- `dataset_name`: Name of the dataset to train on (corresponds to directory in `data/`)
- `fold_n`: Current fold number for cross-validation
- `total_folds`: Total number of folds for cross-validation
- `shuffle_training`: Whether to shuffle the training data
- `batch_size`: Batch size for training
- `n_epochs_per_run`: Number of epochs per training run

## Dataset and Pre-trained Models

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). TEM Mitochondria Segmentation Dataset for Triple Negative Breast Cancer Chemotherapy Analysis (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15602048

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). U-Net Model Weights for TEM Mitochondria Segmentation in Triple Negative Breast Cancer (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15602446

## Citation

TODO

## Acknowledgments

TODO
