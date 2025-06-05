# Data and Models

## Pre-trained Models Download

The pre-trained models are hosted on [Zenodo](https://zenodo.org/records/15602446). We provide a convenience script to download and extract:

```bash
# Navigate to the repository root
cd tem-seg-reproducibility

# Run the download script
bash scripts/download_models.sh
```

This script will download and extract the following files:
- `tem-seg-models_v1.0.0.tar.gz`: Contains all pre-trained models

## Training Data Download

The TEM image dataset and masks are hosted on [Zenodo](https://zenodo.org/records/15602048). We provide a convenience script to download and extract:

```bash
# Navigate to the repository root
cd tem-seg-reproducibility

# Run the download script
bash scripts/download_images.sh
```

This script will download and extract the following files:
- `tem-seg-data_slide_images.tar.gz`: Contains all TEM images
- `tem-seg-data_mitochondria_masks.tar.gz`: Contains all mitochondria masks

## Preparing TFRecords

Before training, you need to convert the images and masks into TFRecord format:

```bash
# Navigate to the repository root
cd tem-seg-reproducibility

# Run the TFRecord creation script
bash scripts/make_tfrecords.sh
```
This script will process the following datasets:
- DRP1-KO
- HCI-010
- Mixture
- PIM001-P

For each dataset, it creates TFRecords for both training/validation and test sets (except for the Mixture dataset which doesn't have a test set).

## Dataset Citation

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). TEM Mitochondria Segmentation Dataset for Triple Negative Breast Cancer Chemotherapy Analysis (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15602048

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). U-Net Model Weights for TEM Mitochondria Segmentation in Triple Negative Breast Cancer (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15602446
