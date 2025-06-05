from .utils import compute_output_size

common_config = dict(
    tile_shape=(444, 444),
    resize_scale=1,
    gamma_correction_peak_target=None,
    smooth_radius=0,
    pad_mode="constant",
    batch_size=128,
    layer_depth=5,
    filters_root=16,
)

config = {
    "cell": {
        **common_config,
        "fraction_of_empty_to_keep": 1.0,
    },
    "mitochondria": {
        **common_config,
        "fraction_of_empty_to_keep": 1.0,
        "target_scale": 0.0075,
    },
    "nucleus": {
        **common_config,
        "fraction_of_empty_to_keep": 1.0,
        "target_scale": 0.03,
    },
}

for organelle, params in config.items():
    params["window_shape"] = (
        compute_output_size(
            input_size=params["tile_shape"][0], layer_depth=params["layer_depth"]
        ),
        compute_output_size(
            input_size=params["tile_shape"][1], layer_depth=params["layer_depth"]
        ),
    )
