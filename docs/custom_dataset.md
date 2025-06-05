# Custom Dataset for Training

## Data Requirements

### Input Images

- TEM images should be in TIFF format (or other formats supported by PIL)
- Masks should be single-channel images (mode 'L' or 'P' in PIL)
- Images should be properly calibrated with scale information displayed on the image. This should work for the images used in this study, but may not work for other images if the scale format is different.
- Images and masks should have matching filenames for proper pairing during preprocessing.

The pipeline expects the following dataset organization:

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

```
data/
└── dataset_name/
    ├── tra_val/organelle/tfrecords/
    │   └── *.tfrecord
    └── tst/organelle/tfrecords/
        └── *.tfrecord
```
