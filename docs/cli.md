# TEM-Seg Command Line Interface

## Overview

The TEM-Seg CLI has been reorganized to support both legacy functionality and new features. The new structure preserves backward compatibility while allowing for the addition of modern components.

## CLI Structure

```
tem-seg
├── v1 (legacy functionality)
│   ├── train
│   ├── predict
│   ├── analyze
│   ├── consolidate
│   └── preprocess
│       └── tfrecords
└── scale
    └── get-scale
```

## Usage

### Legacy Commands (v1)

All existing functionality is available under the `v1` subcommand:

```bash
# Train a model (legacy)
tem-seg v1 train <dataset_name> --organelle mitochondria

# Predict using a model (legacy)
tem-seg v1 predict <image_paths> --model-version Mixture

# Analyze results (legacy)
tem-seg v1 analyze <study_name> --model-name Mixture

# Preprocess data (legacy)
tem-seg v1 preprocess tfrecords <dataset_name> <slides_dir> <masks_dir> --organelle mitochondria
```

### Backward Compatibility

For scripts that might depend on the old command structure, a backward-compatible entry point is available:

```bash
# These are equivalent:
tem-seg-v1 train <dataset_name>
tem-seg v1 train <dataset_name>
```

### New Commands

#### Scale Estimation System

The Scale Estimation System provides tools for detecting scales in TEM images:

```bash
# Detect scale in a single image
tem-seg scale get-scale path/to/image.tif

# Process multiple images with verbose output
tem-seg scale get-scale path/to/image1.tif path/to/image2.tif --verbose

# Save results to CSV file
tem-seg scale get-scale path/to/*.tif --output scales.csv

# Provide fallback scale when detection fails
tem-seg scale get-scale path/to/image.tif --fallback 0.00125
```

## Help Commands

Get help for any command by adding `--help`:

```bash
# Main help
tem-seg --help

# Legacy command help
tem-seg v1 --help
tem-seg v1 train --help

# Scale command help
tem-seg scale --help
tem-seg scale get-scale --help
```

## Technical Notes

- The CLI is implemented using the `typer` library, which builds on top of `click`
- The legacy commands are preserved exactly as they were, just moved under the `v1` namespace
- New commands follow a more modular structure with dedicated modules per feature
