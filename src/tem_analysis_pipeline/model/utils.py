import os
from typing import Tuple
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter


def compute_output_size(input_size=316, layer_depth=5, return_all=False):
    output_size = input_size
    for i in range(layer_depth-1):
        # print(output_size, end=' ')
        output_size -= 4
        assert output_size % 2 == 0, "Found layer with odd size. True again"
        output_size /= 2
    
    mid_size = output_size
    for i in range(layer_depth-1):
        # print(output_size, end=' ')
        output_size -= 4
        output_size *= 2

    output_size -= 4
    # print(output_size)

    if return_all:
        return int(input_size), int(mid_size), int(output_size)
    return int(output_size)


def compute_input_size(output_size=132, layer_depth=5, return_all=False):
    input_size = output_size
    for i in range(layer_depth-1):
        # print(input_size, end=' ')
        input_size += 4
        assert input_size % 2 == 0, "Found layer with odd size. True again"
        input_size /= 2
    
    mid_size = input_size + 4
    for i in range(layer_depth-1):
        # print(input_size, end=' ')
        input_size += 4
        input_size *= 2

    input_size += 4
    # print(input_size)

    if return_all:
        return int(input_size), int(mid_size), int(output_size)
    return int(input_size)


def read_tif_image(image_path, resize_scale=1., gamma_correction_peak_target=None):
    '''Reads an image in tif format and returns a ndarray

    Parameters
    ----------
    image_path: str
        The path to the TIF image
    
    Returns
    -------
    ndarray
        Image as an array object with shape (height, width)
    '''
    # 1. read image from disk
    pil_image = Image.open(image_path)
    assert pil_image.mode in ['L','P','I;16']
    height, width = pil_image.size

    if pil_image.mode != 'L':
        x = np.array(pil_image)
        x = ((x / np.iinfo(x.dtype).max) * 255).round().astype(np.uint8)
        pil_image = Image.fromarray(x)

    # 2. preprocess image
    # 2.1 gamma correction
    if gamma_correction_peak_target is not None:
        a = np.array(pil_image) / 255
        m = a[np.logical_and(a > 0.04, a < 0.60)].mean()
        g = np.log(gamma_correction_peak_target) / np.log(m)
        a = a ** g
        pil_image = Image.fromarray((a * 255).round().astype(np.uint8), 'L')

    # 2.2 rescale
    if resize_scale != 1.:
        new_size = int(height*resize_scale), int(width*resize_scale)
        pil_image = pil_image.resize(new_size, resample=Image.LANCZOS)

    # 3. convert image into ndarray
    image_array = np.array(pil_image)

    return image_array


def read_png_image(image_path, resize_scale=1., smooth_radius = 2):
    '''Reads an image in png format and returns a ndarray

    Parameters
    ----------
    image_path: str
        The path to the PNG image
    
    Returns
    -------
    ndarray
        Image as an array object with shape (height, width)
    '''
    # 1. read image from disk
    pil_image = Image.open(image_path)
    assert pil_image.mode in ['P', 'L']
    height, width = pil_image.size

    # 2. preprocess image
    # 2.1 change mode
    if pil_image.mode == 'P':
        arr_image = np.array(pil_image).astype(float)
        assert arr_image.max() == 1
        pil_image = Image.fromarray((arr_image * 255).astype(np.uint8), mode='L')

    # 2.2 rescale
    if resize_scale != 1.:
        new_size = int(height*resize_scale), int(width*resize_scale)
        pil_image = pil_image.resize(new_size, resample=Image.LANCZOS)
    
    # 2.3 smoothen label edges
    if smooth_radius > 0:
        pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius = smooth_radius))

    # 3. convert image into ndarray
    image_array = np.array(pil_image)
    
    return image_array


def get_tiles_from_array(img_array, tile_shape, stride_shape, window_shape=None, pad_mode='reflect', return_coords=False):
    '''Tile input array

    Parameters
    ----------
    img_array: ndarray
        Array with shape (N, M)
    tile_shape: tuple
        Tuple of len == 2
    stride_shape: tuple
        Tuple of len == 2. Distance between tiles in each direction.
    window_shape: tuple, optional
        Tuple of len == 2. This is the model output shape. This is used to make tiles
        with added borders (burr), so the resulting predictions correspond to the 
        complete input image.
    return_coords: bool, optional
        Whether to return coordinates for each tile

    Returns
    -------
    ndarray or Tuple[ndarray, list]
        An array with shape (n_tiles, tile_height, tile_width)
        if return_coords == True, then a second value is returned with tiles coords
    '''

    if window_shape is None:
        window_shape = tile_shape
    
    # 1. compute coordinates of tiles and store in list
    strd_height, strd_width = stride_shape
    tile_height, tile_width = tile_shape
    wndw_height, wndw_width = window_shape
    
    height, width = img_array.shape
    n_rows = height // strd_height
    n_cols = width  // strd_width
    coord_candidates = [(i * strd_height, j * strd_width) for i in range(n_rows) for j in range(n_cols)]

    # 2. Crop out the parts of image not fitting into tiles
    ver_border, hor_border = (tile_height - wndw_height) // 2, (tile_width - wndw_width) // 2
    height_pad, width_pad = (n_rows - 1) * strd_height + wndw_height - height, (n_cols - 1) * strd_width + wndw_width - width

    extended_img_array = np.pad(img_array, [[ver_border, ver_border+height_pad], [hor_border, hor_border+width_pad]], pad_mode)

    # 4. read slices for tiles from input_array and store in list
    img_tile_list = []
    coord_list = []
    for v_pos, h_pos in coord_candidates:
        img_tile = extended_img_array[v_pos:v_pos+tile_height, h_pos:h_pos+tile_width]
        if img_tile.shape == tile_shape:
            img_tile_list.append(img_tile)
            coord_list.append((v_pos, h_pos))

    # 5. generate final output array
    img_tiled_data = np.array(img_tile_list)

    if return_coords:
        return img_tiled_data, coord_list

    return img_tiled_data


def get_array_from_tiles(tiled_data, coord_list, window_shape=None, output_shape=None):
    '''Stitches tiles together to form the larger array

    This method must figure out the tiling mode by detecting overlap regions.
    For overlaping regions the mean value will be used.

    Parameters
    ----------
    tiled_data: ndarray
        An array with shape (n_tiles, tile_height, tile_width, n_channels)
    coord_list: list
        A list with corresponding coordinates for each tile. coord = (d0, d1)
    window_shape: tuple, optional
        Tuple of len == 2. This is the model output shape. This is used to make tiles
        with added borders (burr), so the resulting predictions correspond to the 
        complete input image.
    output_shape: tuple, optional
        Tuple of len == 2. After assembling image from tiles, extra pixels may be
        present at the right and bottom. By providing the output_shape parameter,
        we can return the image cropped to the correct shape.

    Returns
    -------
    ndarray
        The larger stitched array
    '''
    # 0. prepare for overlaping areas which will likely overflow
    dtype = tiled_data.dtype
    tiled_data = tiled_data.astype(float)
    tile_shape = tiled_data[0].shape
    if window_shape is None:
        window_shape = tile_shape

    # 1. figure out ...
    tile_height, tile_width = tile_shape
    wndw_height, wndw_width = window_shape
    v_border, h_border = (tile_height - wndw_height)//2, (tile_width - wndw_width)//2
    d0_coords, d1_coords = zip(*coord_list)
    height, width = max(d0_coords) + tile_height, max(d1_coords) + tile_width

    
    # 2. initialize output array
    output_array = np.zeros((height, width), dtype=tiled_data.dtype)
    data_count = np.zeros_like(output_array)

    # 3. write tile data into output array
    for i, (ver_pos, hor_pos) in enumerate(coord_list):
        output_array[ver_pos:ver_pos+tile_height, hor_pos:hor_pos+tile_width] += tiled_data[i]
        data_count[ver_pos:ver_pos+tile_height, hor_pos:hor_pos+tile_width] += 1

    output_array[output_array>0] = output_array[output_array>0] / data_count[output_array>0]

    if v_border:
        output_array = output_array[v_border:-v_border, :]
    if h_border:
        output_array = output_array[:, v_border:-v_border]

    if output_shape is not None:
        if output_array.shape[0] >= output_shape[0]:
            output_array = output_array[:output_shape[0], :]
        else:
            output_array = np.pad(output_array, [[0, output_shape[0] - output_array.shape[0]] ,[0, 0]])

        if output_array.shape[1] >= output_shape[1]:
            output_array = output_array[:, :output_shape[1]]
        else:
            output_array = np.pad(output_array, [[0, 0], [0, output_shape[1] - output_array.shape[1]]])

    return output_array.astype(dtype)


def augment_tile_data(tiled_data: np.ndarray, times: int) -> np.ndarray:
    '''Applies transformations to input and returns the concatenation of these

    Parameters
    ----------
    tiled_data: ndarray
        Array with shape (n_tiles, height, width)
    times: int
        Multiplication factor for the augmentation process. This value must be
        in the range [1, 4]
    
    Returns
    -------
    augmented_data: ndarray
        The array of concatenated transformations with original input
    '''
    assert times >= 1 and times <= 4
    
    transformations = []

    for i in range(times):
        x = np.rot90(tiled_data, k=i*2, axes=(1, 2))
        if i//2 % 2 == 1:
            x = x[:, :, ::-1]
        transformations.append(x)

    return np.concatenate(transformations)


def _make_bytes_feature(tile):

    tile = tf.expand_dims(tile, 2)

    tile_serialized = tf.io.serialize_tensor(tile)
    # tile_serialized = tf.io.encode_png(tile)

    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tile_serialized.numpy()]))


def _make_example(record: 'tuple[tf.Tensor, tf.Tensor]') -> tf.train.Example:

    img_tile, msk_tile = record
    img_tile_feature_of_bytes = _make_bytes_feature(img_tile)
    msk_tile_feature_of_bytes = _make_bytes_feature(msk_tile)

    features_for_example = {
        'img_tile': img_tile_feature_of_bytes,
        'msk_tile': msk_tile_feature_of_bytes, }

    return tf.train.Example(
        features=tf.train.Features(feature=features_for_example))


def write_tfrecord_for_image_mask_pair( image_path: str, tile_shape: Tuple[int, int], layer_depth: int,
                                        resize_scale: float=1., gamma_correction_peak_target: float=0.45,
                                        smooth_radius: int=2, pad_mode: str = 'constant', **kwargs):
    dirpath, image_filename = os.path.split(image_path)

    filename = os.path.splitext(image_filename)[0]

    label_path = os.path.join(dirpath, filename + '-labels.png')

    # we will assume that the label png is in the same folder as the tif image
    img_array = read_tif_image(image_path, resize_scale=resize_scale, gamma_correction_peak_target=gamma_correction_peak_target)
    msk_array = read_png_image(label_path, resize_scale=resize_scale, smooth_radius=smooth_radius)

    # this is the size of the window at the output layer
    window_shape = (compute_output_size(input_size=tile_shape[0], layer_depth=layer_depth),
                    compute_output_size(input_size=tile_shape[1], layer_depth=layer_depth))

    # stride should be smaller than or equal the output, so we make sure not to skip pixels between tiles
    # we also try to choose a stride such that pixels at the end are not excluded
    stride_shape = list(int(img_array.shape[i] * resize_scale) for i in range(2))
    for i in range(2):
        while stride_shape[i] > window_shape[i]:
            stride_shape[i] //= 2
    stride_shape = tuple(stride_shape)

    image = get_tiles_from_array(img_array, tile_shape, stride_shape, window_shape, pad_mode=pad_mode) / 255
    label = get_tiles_from_array(msk_array, tile_shape, stride_shape, window_shape, pad_mode=pad_mode) / 255
    image = image.astype(np.float32)
    label = label.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((image, label))

    options = tf.io.TFRecordOptions(compression_type='')
    output_dirpath = os.path.join(dirpath, f'{tile_shape[0]}by{tile_shape[1]}_tiles')
    os.makedirs(output_dirpath, exist_ok=True)
    output_filepath = os.path.join(output_dirpath, filename + '.tfrecord')
    with tf.io.TFRecordWriter(output_filepath, options=options) as file_writer:
        for record in dataset:
            example = _make_example(record)
            record_bytes = example.SerializeToString()
            file_writer.write(record_bytes)


def read_tfrecord(record):

    # as defined in _make_example
    features_for_example = {
        'img_tile': tf.io.FixedLenFeature([], tf.string),
        'msk_tile': tf.io.FixedLenFeature([], tf.string), }
    
    # read the sample
    sample =  tf.io.parse_single_example(record, features_for_example)

    # arrays were created and stored as tf.float32
    img_tile = tf.io.parse_tensor(sample['img_tile'], tf.float32)
    msk_tile = tf.io.parse_tensor(sample['msk_tile'], tf.float32)

    # # cast to float and normalize
    # img_tile = tf.cast(img_tile, tf.float32) / 255
    # msk_tile = tf.cast(msk_tile, tf.float32) / 255
    # msk_tile_1 = tf.cast(msk_tile, tf.float32) / 255
    # msk_tile_0 = 1. - msk_tile_1
    # msk_tile = tf.concat([msk_tile_0, msk_tile_1], axis=-1)

    return img_tile, msk_tile


def set_tile_shape(tile_shape):
    def set_shapes(image, label):
        image.set_shape(tile_shape + (1,))
        label.set_shape(tile_shape + (1,))
        return image, label
    return set_shapes


def filter_empty_labels(_, label):
    return tf.math.reduce_any(label > 0)


def keep_fraction_of_empty_labels(fraction):
    def filter_function(image, label):
        # want to take a fraction of samples with no label in it
        # using random numbers is problematic since it will likely result in epochs with 
        # different number of batches.
        # hence, I will use a the following approach that just seems random

        # this (reduce_sum % 1) is a number between 0 and 1, it depends only on the image values
        return tf.math.reduce_any(label > 0) or tf.math.reduce_sum(image) / 100 % 1 < fraction

    if not fraction > 0.:
        return filter_empty_labels
    elif fraction < 1.:
        return filter_function
    else:
        return lambda image, label: True


def to_numpy_or_python_type(tensors):
    """Converts a structure of `Tensor`s to `NumPy` arrays or Python scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to deal
    with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Args:
        tensors: A structure of tensors.
    Returns:
        `tensors`, but scalar tensors are converted to Python types and non-scalar
        tensors are converted to Numpy arrays.
    """
    def _to_single_numpy_or_python_type(t):
        if isinstance(t, tf.Tensor):
            x = t.numpy()
            return x.item() if np.ndim(x) == 0 else x
        return t  # Don't turn ragged or sparse tensors to NumPy.

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)


def random_flip_and_rotation(image, label):

    r = tf.random.uniform([2], maxval=4, dtype=tf.int32)
    k = r[0]
    f = r[1]
    if image.shape[0] != image.shape[1]:
        # rectangular image. make sure to keep aspect ratio
        k = 2*k % 4

    if k != 0:
        image = tf.image.rot90(image, k=k)
        label = tf.image.rot90(label, k=k)

    if f % 2 == 1:
        image = tf.image.flip_left_right(image)
        label = tf.image.flip_left_right(label)

    return image, label


def random_image_adjust(image, label):

    height, width, _ = image.shape

    r = tf.random.normal([4], [1, 1, 0, 1], [0.08, 0.03, 0.03, 0.15])
    g = r[0]
    c = r[1]
    b = r[2]
    s = r[3]

    image = tf.image.adjust_gamma(image, g)
    image = tf.image.adjust_contrast(image, c)
    image = tf.image.adjust_brightness(image, b)

    new_height = int(s * height)
    new_width = int(s * width)
    image = tf.image.resize(image, [new_height, new_width], method='bicubic', preserve_aspect_ratio=True, antialias=False)
    image = tf.image.resize_with_crop_or_pad(image, height, width)
    label = tf.image.resize(label, [new_height, new_width], method='nearest', preserve_aspect_ratio=True)
    label = tf.image.resize_with_crop_or_pad(label, height, width)

    image = tf.clip_by_value(image, 0, 1)
    label = tf.clip_by_value(label, 0, 1)

    return image, label


# taken from site-packages/unet
def crop_to_shape(data, shape: Tuple[int, int, int]):
    """
    Crops the array to the given image shape by removing the border

    :param data: the array to crop, expects a tensor of shape [batches, nx, ny, channels]
    :param shape: the target shape [batches, nx, ny, channels]
    """
    diff_nx = (data.shape[0] - shape[0])
    diff_ny = (data.shape[1] - shape[1])

    if diff_nx == 0 and diff_ny == 0:
        return data

    offset_nx_left = diff_nx // 2
    offset_nx_right = diff_nx - offset_nx_left
    offset_ny_left = diff_ny // 2
    offset_ny_right = diff_ny - offset_ny_left

    cropped = data[offset_nx_left:(-offset_nx_right), offset_ny_left:(-offset_ny_right)]

    assert cropped.shape[0] == shape[0]
    assert cropped.shape[1] == shape[1]
    return cropped


# taken from site-packages/unet
def crop_labels_to_shape(shape: Tuple[int, int, int]):
    def crop(image, label):
        return image, crop_to_shape(label, shape)
    return crop

