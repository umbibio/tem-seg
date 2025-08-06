from pathlib import Path

import tensorflow as tf


def inspect_tfrecord_dtype(filepath: str | Path | list[str]):
    """
    Inspect a TFRecord file to determine the dtype of stored tensors.

    Args:
        filepath: Path to the TFRecord file
        check_gzip: Whether to check if the file is gzipped

    Returns:
        dict: Information about the dtypes found in the file
    """
    if isinstance(filepath, list):
        for fp in filepath:
            print(fp)
            inspect_tfrecord_dtype(fp)

        return

    filepath = Path(filepath)

    # Check if file is gzipped
    compression_type = None
    with open(filepath, "rb") as f:
        if f.read(2) == b"\x1f\x8b":
            compression_type = "GZIP"

    # Create dataset for single file
    dataset = tf.data.TFRecordDataset(str(filepath), compression_type=compression_type)

    # Define feature description for parsing
    features = {
        "img_tile": tf.io.FixedLenFeature([], tf.string),
        "msk_tile": tf.io.FixedLenFeature([], tf.string),
    }

    # Try to read first record
    for raw_record in dataset.take(1):
        sample = tf.io.parse_single_example(raw_record, features)

        # Try different dtypes to see which one works
        dtypes_to_try = [
            tf.float16,
            tf.float32,
            tf.float64,
            tf.int32,
            tf.int64,
            tf.uint8,
        ]

        for key in ["img_tile", "msk_tile"]:
            tensor_string = sample[key]

            result = None
            for dtype in dtypes_to_try:
                try:
                    parsed = tf.io.parse_tensor(tensor_string, dtype)
                    # If successful, record the dtype and shape
                    result = {
                        "dtype": dtype.name,
                        "shape": parsed.shape.as_list(),
                        "sample_values": parsed.numpy()
                        .flatten()[:5]
                        .tolist(),  # First 5 values
                    }
                    break
                except Exception:
                    print(f"  {key}: ✗ Failed to parse as {dtype.name}")
                    continue

            if result is None:
                print(f"  ⚠ Could not determine dtype for {key}")
                # Try to get raw info about the tensor
                try:
                    # Get the raw bytes to inspect
                    raw_bytes = tensor_string.numpy()
                    print(f"    Raw string length: {len(raw_bytes)} bytes")
                except Exception as e:
                    print(f"    Could not inspect raw bytes: {e}")
