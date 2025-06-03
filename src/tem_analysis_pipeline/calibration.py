from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


class NoScaleError(Exception): ...


class NoScaleNumberError(Exception): ...


class NoScaleUnitError(Exception): ...


def text_phantom(text):
    # based on SOF
    # https://stackoverflow.com/questions/45947608/rendering-a-unicode-ascii-character-to-a-numpy-array

    fontdir = Path(__file__).parent / "fonts"

    # meant to create templates for single letters
    assert len(text) == 1, "`text_phantom` takes a single character"
    size = 430
    fontfile = fontdir / "LiberationSans-Bold.ttf"

    # Create font
    pil_font = ImageFont.truetype(fontfile, size=size, encoding="unic")
    bbox = pil_font.getbbox(text)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]

    # create a blank canvas
    canvas = Image.new("L", [text_width * 2, text_height * 2], (0,))

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = (0, 0)

    draw.text(offset, text, font=pil_font, fill=255)

    arr = np.asarray(canvas)
    arr = crop(arr)

    return arr


def crop(a):
    vl, vh = np.argwhere(np.any(a != 0, axis=1))[[0, -1], 0]
    hl, hh = np.argwhere(np.any(a != 0, axis=0))[[0, -1], 0]
    vl, vh = np.clip([vl - 1, vh + 1], 0, 100_000)
    hl, hh = np.clip([hl - 1, hh + 1], 0, 100_000)
    return a[vl:vh, hl:hh]


def crop_and_stretch(a, size):
    return cv2.resize(crop(a), (size, size))


CHARACTER_TEMPLATE_ALPHABET = "0123456789umnμ"


def get_character_templates(size=50):
    nclasses = len(CHARACTER_TEMPLATE_ALPHABET)

    M = np.matrix(np.zeros((nclasses, size * size)))
    for i, c in enumerate(CHARACTER_TEMPLATE_ALPHABET):
        a = text_phantom(c)
        a = crop_and_stretch(a, size)
        M[i] = a.flatten() / 255

    return M


CHARACTER_TEMPLATE_SIZE = 50
CHARACTER_TEMPLATE = get_character_templates(CHARACTER_TEMPLATE_SIZE)


def predict_character(a):
    a = np.matrix(crop_and_stretch(a, CHARACTER_TEMPLATE_SIZE).flatten()) / 255
    return CHARACTER_TEMPLATE_ALPHABET[
        np.argmin(np.abs(CHARACTER_TEMPLATE - a).sum(axis=1))
    ]


def get_calibration(img):
    if isinstance(img, Image.Image):
        assert img.mode == "L"
        img = np.array(img)

    black_rw_fr = (img[-200:] == 0).all(axis=1).sum() / 200
    white_rw_fr = (img[-200:] == np.iinfo(img.dtype).max).all(axis=1).sum() / 200

    if black_rw_fr > 0.001 and black_rw_fr > white_rw_fr:
        # print("try cal 01")
        return get_calibration_01(img)

    if white_rw_fr > 0.001 and white_rw_fr > black_rw_fr:
        # print("try cal 02")
        return get_calibration_02(img)

    # print("try cal 03")
    return get_calibration_03(img)


def get_calibration_03(img):
    nrow, ncol = img.shape
    stroke_size = 16
    yptr = nrow - stroke_size - 1
    found_rect = False
    while not found_rect and yptr > 0:
        while (
            np.all(img[yptr : yptr + stroke_size] == 0, axis=0).sum() == 0 and yptr > 0
        ):
            yptr -= 1
        bar_yend = yptr + stroke_size
        bar_ystart = bar_yend - stroke_size

        xptr = 0
        while not found_rect and xptr < ncol - 1:
            awh = np.argwhere(np.all(img[bar_ystart:bar_yend, xptr:] == 0, axis=0))
            if len(awh) == 0:
                break
            bar_xstart = awh[0, 0] + xptr
            bar_xend = (
                np.argwhere(np.any(img[bar_ystart:bar_yend, bar_xstart:] != 0, axis=0))[
                    0, 0
                ]
                + bar_xstart
            )

            tst1 = (img[bar_ystart:bar_yend, bar_xstart - 1] > 0).sum() >= 5
            tst2 = (img[bar_ystart:bar_yend, bar_xend] > 0).sum() >= 5
            tst3 = (img[bar_ystart - 1, bar_xstart:bar_xend] > 0).sum() >= (
                bar_xend - bar_xstart
            ) // 3
            tst4 = (img[bar_yend, bar_xstart:bar_xend] > 0).sum() >= (
                bar_xend - bar_xstart
            ) // 3

            found_rect = tst1 and tst2 and tst3 and tst4
            xptr = bar_xend
        yptr -= 1

    if not found_rect:
        raise NoScaleError("Couldn't find scale bar v3")

    scale_px = bar_xend - bar_xstart

    text_yend = bar_ystart - 1
    yptr = text_yend - 4 * stroke_size
    yptr -= np.argwhere(
        (np.sum(img[:yptr, bar_xstart:bar_xend] == 0, axis=1) < stroke_size // 2)[::-1]
    )[0, 0]
    text_ystart = yptr - 1

    text_img = img[text_ystart:text_yend, bar_xstart:]
    text_img = (text_img == 0).astype(np.uint8) * 255

    scale_um = read_scale_um(text_img)
    um_per_pixel = scale_um / scale_px

    return um_per_pixel


def get_calibration_02(arr: NDArray):
    nrow, ncol = arr.shape
    band_height = np.argwhere(arr[::-1, ncol // 2] != 255)[0, 0]
    print(f"{band_height=}")
    roi = arr[-band_height:, ncol // 2 :]
    yptr = np.argwhere(np.any(roi != 255, axis=1))[0, 0]
    value = 0
    count = 0
    while np.any(roi[yptr] != 255):
        value += np.sum(roi[yptr] == 0)
        count += 1
        yptr += 1
    scale_px = value / count
    print(f"{scale_px=}")

    try:
        roi = roi[yptr:]
        yptr = np.argwhere(np.any(roi != 255, axis=1))[0, 0]
        roi = roi[yptr:]
        yptr = np.argwhere(np.all(roi == 255, axis=1))[0, 0]
        roi = roi[:yptr]
        xlptr = np.argwhere(np.any(roi != 255, axis=0))[0, 0]
        roi = roi[:, xlptr:]
        text_img = roi
    except IndexError as e:
        raise NoScaleError(f"Couldn't find scale bar v2: {e}")

    scale_um = read_scale_um(255 - text_img)
    um_per_pixel = scale_um / scale_px

    return um_per_pixel


def get_calibration_01(img):
    scale_end = np.argwhere(np.any(img[::-1] == 255, axis=1))[0, 0]
    scale_height = (
        np.argwhere(np.all(img[-scale_end - 1 :: -1] == 0, axis=1))[0, 0] + scale_end
    )
    bar_start, text_end = np.argwhere(np.any(img[-scale_height:] == 255, axis=0))[
        [0, -1], 0
    ]
    bar_end = (
        np.argwhere(np.all(img[-scale_height:, bar_start:] == 0, axis=0))[0, 0]
        + bar_start
    )
    num_start = (
        np.argwhere(np.any(img[-scale_height:, bar_end:] == 255, axis=0))[0, 0]
        + bar_end
    )
    bar_outer_length = bar_end - bar_start
    x = np.sum(
        img[-scale_height:-scale_end, bar_start + bar_outer_length // 2 : bar_end]
        == 255,
        axis=1,
    )
    line_width = int(x[(x < bar_outer_length // 2 - 1) * (x > 0)].mean().round())
    scale_px = bar_outer_length - line_width

    text_img = img[-scale_height:-scale_end, num_start - 2 : text_end + 2]

    scale_um = read_scale_um(text_img)
    um_per_pixel = scale_um / scale_px

    return um_per_pixel


def read_scale_um(text_img):
    # Image.fromarray(text_img).save("text_img.png")
    text_height = text_img.shape[0]

    nmbr_string = ""
    unit = ""
    i = 0
    while not np.all(text_img[:, : text_height * 2] == 0):
        i += 1
        j = np.argwhere(np.any(text_img == 255, axis=0))[0, 0]
        text_img = text_img[:, j:]
        r = np.argwhere(np.all(text_img == 0, axis=0))[0, 0]
        a = text_img[:, :r]
        # Image.fromarray(a).save(f"text_img_{i}.png")
        c = predict_character(a)
        if c.isnumeric():
            nmbr_string += c
        elif c in "nuμm":
            unit += c

        if len(unit) == 2:
            break

        text_img = text_img[:, r:]
    print("Slide scale is:", nmbr_string, unit)
    if len(nmbr_string) == 0:
        raise NoScaleNumberError("Could not read the number in scale")
    nmbr = float(nmbr_string)

    if unit in ["um", "μm"]:
        scale_um = nmbr
    elif unit == "nm":
        scale_um = nmbr / 1000
    elif unit == "mm":
        scale_um = nmbr * 1000
    else:
        raise NoScaleUnitError("Didn't recognize the scale units")

    return scale_um


def _fix_image_scale(
    img: Image.Image, trg_scale: float, img_scale: float, resample: Image.Resampling
):
    scale_ratio = trg_scale / img_scale
    trg_width = int(np.round(img.width / scale_ratio))
    trg_height = int(np.round(img.height / scale_ratio))
    trg_size = (trg_width, trg_height)

    return img.resize(trg_size, resample=resample)


def fix_image_scale(img: Image.Image, trg_scale: float, img_scale: float = None):
    if img_scale is None:
        img_scale = get_calibration(img)
        print(f"Detected image scale: {img_scale}")

    return _fix_image_scale(img, trg_scale, img_scale, Image.Resampling.LANCZOS)


def fix_mask_scale(msk: Image.Image, trg_scale: float, img_scale: float):
    return _fix_image_scale(msk, trg_scale, img_scale, Image.Resampling.NEAREST)
