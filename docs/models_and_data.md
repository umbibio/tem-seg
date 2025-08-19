# Data and Models

## Pre-trained Models and Training Data

The CLI can automatically download assets as needed (e.g., model weights at prediction time). If you plan to train or fine‑tune models, you can fetch datasets and pre‑trained weights from Zenodo using our convenience command.

Resources available:
- Slide images
- Mitochondria semantic segmentation masks
- Pre-trained model weights

Example usage:
```bash
tem-seg download --help
tem-seg download model_weights
```
Use the help output to see all available resources and their identifiers.

When downloading model weights, this will retrieve and extract the archive containing all pre-trained models:
- `tem-seg-models_v#.#.#.tar.gz`

## Models Included

Pre-trained models included for mitochondria segmentation:
- DRP1-KO
- HCI-010
- Mixture
- PIM001-P

## Preparing TFRecords

Before training, you need to convert the images and masks into TFRecord format:

```bash
# Navigate to the repository root
cd tem-seg

# Run the TFRecord creation script
bash scripts/make_tfrecords.sh
```
This script will process the following datasets:
- DRP1-KO
- HCI-010
- Mixture
- PIM001-P

For each dataset, it creates TFRecords for both training/validation and test sets (except for the Mixture dataset which doesn't have a test set).

On Windows, you can use the provided Command Prompt or PowerShell variants:

```
:: Windows (Command Prompt)
scripts\make_tfrecords.bat
```

```powershell
# Windows (PowerShell)
pwsh -File scripts/make_tfrecords.ps1
```

## Dataset Citation

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). TEM Mitochondria Segmentation Dataset for Triple Negative Breast Cancer Chemotherapy Analysis (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15602048

Arriojas Maldonado, A. A., Baek, M., Berner, M. J., Zhurkevich, A., Hinton, Jr., A., Meyer, M., Dobrolecki, L., Lewis, M. T., Zarringhalam, K., & Echeverria, G. (2025). U-Net Model Weights for TEM Mitochondria Segmentation in Triple Negative Breast Cancer (1.0.0). Zenodo. https://doi.org/10.5281/zenodo.15602446
