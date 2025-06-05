# TEM Analysis Pipeline

## Overview

This repository contains a pipeline for the analysis of mitochondria morphology using semantic segmentation on Transmission Electron Microscopy (TEM) images. The pipeline includes tools for preprocessing TEM images, training U-Net models for semantic segmentation, generating predictions on new images, and analyzing organelle morphology.

## Documentation

- [Installation](docs/installation.md)
- [Data and Models](docs/data.md)
- [Custom Dataset](docs/custom_dataset.md)
- [Usage Guide](docs/usage.md)
- [Configuration](docs/configuration.md)

# Installation

## Requirements

- Python 3.12
- TensorFlow 2.19

## Setup

Clone the repository and install the package:

```bash
# Clone the repository
git clone https://github.com/umbibio/tem-seg-reproducibility.git
cd tem-seg-reproducibility

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package in development mode
pip install -e .
```

This will install the `tem-seg` command-line tool and its dependencies.

## Pre-trained Models Download

The pre-trained models are hosted on Zenodo. You can download them using the provided script:

```bash
# Navigate to the repository root
cd tem-seg-reproducibility

# Run the download script
bash scripts/download_models.sh
```

This script will download and extract the following archive:
- `tem-seg-models_v1.0.0.tar.gz`: Contains all pre-trained models

Models included for mitochondria segmentation:
- DRP1-KO
- HCI-010
- Mixture
- PIM001-P

See the [usage guide](docs/usage.md) for more information on how to perform predictions and analysis of mitochondria morphology.

## Dataset and Pre-trained Models

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). TEM Mitochondria Segmentation Dataset for Triple Negative Breast Cancer Chemotherapy Analysis (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15602048

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). U-Net Model Weights for TEM Mitochondria Segmentation in Triple Negative Breast Cancer (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15602446

## Citation

TODO

## Acknowledgments

TODO
