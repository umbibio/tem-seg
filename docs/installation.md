# Installation

## Requirements

- Python 3.12
- TensorFlow >= 2.16

## Quickstart

Create a new Python environment following the instructions [below](#environment-setup).

Install the `tem-seg` Python package using the following command:
```bash
pip install git+https://github.com/umbibio/tem-seg.git
```

On Linux, you might want to install CUDA and cuDNN dependencies if you plan to use GPU acceleration.
```bash
pip install tensorflow[and-cuda]
```

Windows users should use WSL2 if possible. Tensorflow GPU support is no longer available on Windows natively.

For GPU acceleration on a macOS system, you might want to install the `tensorflow-metal` package.
```bash
pip install tensorflow-metal
```

These are defined as optional dependencies for `tem-seg` and can be automatically installed with the following commands.
```bash
pip install "tem-seg[cuda] @ git+https://github.com/umbibio/tem-seg.git"
pip install "tem-seg[gpu-mac] @ git+https://github.com/umbibio/tem-seg.git"
```

By remotely installing the `tem-seg` package, you can use the `tem-seg` command-line tool without having to clone the repository.

```bash
tem-seg --help
```

## Environment Setup

It is always recommended to create a virtual environment for new projects. To do so, you may use any suitable tool such as [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html) or [uv](https://uv.readthedocs.io/en/latest/).

### Conda

The conda tool can be installed in multiple ways. Please follow any of these guides:

- [Miniforge](https://github.com/conda-forge/miniforge) (recommended) - a minimal installer for conda, similar to miniconda, that uses the conda-forge channel by default.
- [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) - a miniature installation of Anaconda Distribution that includes only conda, Python, the packages they both depend on, and a small number of other useful packages. *Please note that this tool might require special licensing*.
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) - a full distribution of Python and its packages, with a convenient graphical user interface. *Please note that this tool might require special licensing*.

After installing conda, you can create a new environment using the following command:
```bash
# create the environment
conda create -n tem-seg-env python=3.12
# activate the environment
conda activate tem-seg-env
```

You can finally install new packages using either `pip` or `conda`. It is preferable to use `conda` to install packages within a conda environment.

```bash
conda install package_name
```
or
```bash
pip install package_name
```

### UV package manager

UV is a quite fast Python package and project manager. It makes it painless to maintain multiple Python environments as required by different projects.

- [Install UV](https://docs.astral.sh/uv/getting-started/installation/)

After installing uv, you can use one of the following commands within the cloned repository to fully setup a new environment:
- `uv sync`: Install all the required dependencies
- `uv sync --extra cuda`: Install the required and optional CUDA dependencies
- `uv sync --extra gpu-mac`: Install the required and optional macOS GPU dependencies

You can also manually create a new environment using the following command:

1. Create the environment
```
uv venv --python 3.12
```
2. And to activate the environment
    - On Windows: `.venv/Scripts/activate`
    - On Linux/Mac: `source .venv/bin/activate`

3. And finally install new packages
```bash
uv pip install package_name
```


## Advanced Setup

Clone the repository and install the package:

```bash
# Clone the repository
git clone https://github.com/umbibio/tem-seg.git
cd tem-seg

# Create a virtual environment
# uv
uv sync --extra cuda
# conda
conda env create -f environment.yml
conda activate tem-seg-env

# Install the package in development mode
pip install -e .
```

This will install the `tem-seg` command-line tool and its dependencies.

## Resources Available for Download

The provided CLI will automatically download assets that it might need to run, e.g. specific model weights when running predictions on images.

However, if you wish to train new models or fine-tune existing ones, training data and pre-trained models are made available on Zenodo. We provide a convenience command to download the following resources:

- Slide images
- Mitochondria semantic segmentation masks
- Pre-trained models weights

```bash
tem-seg download --help
tem-seg download model_weights
```

This command will download and extract the following archive:
- `tem-seg-models_v#.#.#.tar.gz`: Contains all pre-trained models

Models included for mitochondria segmentation:
- DRP1-KO
- HCI-010
- Mixture
- PIM001-P

See the [usage guide](usage.md) for more information on how to perform predictions and analysis of mitochondria morphology.

