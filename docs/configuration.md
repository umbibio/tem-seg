# Configuration

## Model Configuration

Model and training parameters are configured in `src/tem_seg/config/_config.py` and `src/tem_seg/settings.toml`. Key parameters include:

- `tile_shape`: Input image size for the model
- `window_shape`: Output mask size for the model. Must match the output shape of the model given the `tile_shape` and `layer_depth` parameters. It is computed automatically in the code.
- `target_scale`: Target magnification scale in um/px for training models. Images will be scaled to this magnification before training/prediction.
- `fraction_of_empty_to_keep`: Fraction of empty masks to keep during training. This can be used to balance the number of empty and non-empty masks in the training data.

These parameters can be customized for different organelles (cell, mitochondria, nucleus).

Example configuration:

```python
config = {
    "mitochondria": {
        "tile_shape": (444, 444),
        "window_shape": (260, 260),
        "target_scale": 0.0075,  # um/px
        "fraction_of_empty_to_keep": 1.0, # Keep all empty masks
    },
    ...
}
```
