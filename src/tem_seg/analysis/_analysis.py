import os
import typing
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon

from .. import prediction_tools

NONDIM_MEASUREMENTS = ["idx", "closest_nuc_idx"]
TWODIM_MEASUREMENTS = ["area", "hull_area"]
UNIDIM_MEASUREMENTS = [
    "arc_length",
    "thickness",
    "extended_length",
    "length",
    "width",
    "min_caliper",
    "max_caliper",
    "mean_caliper",
    "weighted_mean_caliper",
    "closest_nuc_distance",
]

MEASUREMENT_UNITS = {
    **{k: "" for k in NONDIM_MEASUREMENTS},
    **{k: "μm" for k in UNIDIM_MEASUREMENTS},
    **{k: "μm2" for k in TWODIM_MEASUREMENTS},
}


def compute_feret(hull_points):
    # we assume hull points are sorted

    hull_points = hull_points.squeeze()
    if all(hull_points[0] == hull_points[-1]):
        hull_points = hull_points[:-1]

    n = len(hull_points)
    if n < 3:
        return 0.0, 0.0

    x = np.sqrt(((hull_points[None, ...] - hull_points[:, None]) ** 2).sum(axis=-1))

    heights = []
    bases = []
    for i in range(n):
        j = (i + 1) % n
        b = x[i, j]
        h = -1
        ht = 0
        for k in range(i + 2, i + n):
            k = k % n
            a, c = x[i, k], x[j, k]
            s = (a + b + c) / 2
            ht = 2 * np.sqrt(s * (s - a) * (s - b) * (s - c)) / b
            if ht <= h:
                break
            h = ht
        heights.append(h)
        bases.append(b)

    heights = np.array(heights)
    bases = np.array(bases)

    return min(heights), x.max(), heights.mean(), (heights * bases).sum() / bases.sum()


def analyze_organelle_prediction(
    img: Image.Image,
    prd: Image.Image,
    img_scale: float,
    organelle: str,
) -> dict:
    img_filename = os.path.basename(img.filename)
    um_per_pixel = img_scale
    assert um_per_pixel > 0.0, f"{um_per_pixel = }"
    units = "μm"

    prd_arr = np.array(prd)

    # print("\t\tfinding contours", flush=True)
    org_contours = cv2.findContours(prd_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[
        0
    ]

    annotations = []
    # print("\t\tprocesing mitochondria measurements", flush=True)
    org_data_list = []
    for org_idx, org_contour in enumerate(org_contours):
        org_center_um = org_contour.mean(axis=0).round().astype(int)[0] * um_per_pixel
        org_arcl_um = cv2.arcLength(org_contour, True) * um_per_pixel
        org_area_px2 = cv2.contourArea(org_contour)
        org_area_um2 = org_area_px2 * um_per_pixel**2
        _, (w, h), _ = cv2.minAreaRect(org_contour)
        org_length_um = max(w, h) * um_per_pixel
        org_width_um = min(w, h) * um_per_pixel

        if org_area_px2 < 25:
            continue

        org_hull = cv2.convexHull(org_contour)
        (
            org_mincaliper_px,
            org_maxcaliper_px,
            org_meancaliper_px,
            org_wmeancaliper_px,
        ) = compute_feret(org_hull)
        org_mincaliper_um = org_mincaliper_px * um_per_pixel
        org_maxcaliper_um = org_maxcaliper_px * um_per_pixel
        org_meancaliper_um = org_meancaliper_px * um_per_pixel
        org_wmeancaliper_um = org_wmeancaliper_px * um_per_pixel

        # estimating organelle thickness
        x = np.sqrt(
            ((org_contour[None, ...] - org_contour[:, None]) ** 2).sum(axis=-1)
        )[..., 0]
        x = np.tril(x, k=0)
        z = x[x > 0]
        y, bins = np.histogram(z, bins=np.arange(0, int(z.max())))
        org_thickness_um = np.argmax(y) * um_per_pixel

        tips_area = np.pi * (org_thickness_um / 2) ** 2
        inner_area = np.maximum(0, org_area_um2 - tips_area)
        extended_length_um = inner_area / org_thickness_um + org_thickness_um

        org_data = dict(
            sample=img_filename,
            idx=org_idx,
            center=org_center_um.tolist(),
            arc_length=org_arcl_um,
            area=org_area_um2,
            thickness=org_thickness_um,
            extended_length=extended_length_um,
            length=org_length_um,
            width=org_width_um,
            min_caliper=org_mincaliper_um,
            max_caliper=org_maxcaliper_um,
            mean_caliper=org_meancaliper_um,
            weighted_mean_caliper=org_wmeancaliper_um,
        )
        org_data_list.append(org_data)
        org_measurements = [
            {
                "name": (
                    f"{' '.join([kk.capitalize() for kk in k.split('_')])}"
                    f"{' ' + units if k in UNIDIM_MEASUREMENTS else (' ' + units + '^2' if k in TWODIM_MEASUREMENTS else '')}"
                ),
                "value": v,
            }
            for k, v in org_data.items()
            if k in NONDIM_MEASUREMENTS + TWODIM_MEASUREMENTS + UNIDIM_MEASUREMENTS
        ]

        org_contour_coordinates = org_contour.transpose([1, 0, 2]).tolist()
        org_contour_coordinates[0].append(org_contour_coordinates[0][0])
        annotations.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": org_contour_coordinates,
                },
                "properties": {
                    "object_type": "annotation",
                    "classification": {
                        "name": organelle,
                        "colorRGB": {
                            "mitochondria": -16744448,
                            "nucleus": -16711936,
                        }.get(organelle, -16776961),
                    },
                    "isLocked": False,
                    "measurements": org_measurements,
                },
            },
        )

    data = {
        organelle: org_data_list,
        "metadata": dict(
            image_path=img.filename,
            sample_name=img_filename,
            um_per_pixel=um_per_pixel,
        ),
    }

    return data, annotations


def get_labels_and_scores(msk: Image.Image, prd: Image.Image) -> Tuple[list, list]:
    """Compute the labels and scores for the ground truth and predicted masks.

    Parameters
    ----------
    msk : Image.Image
        ground truth mask
    prd : Image.Image
        predicted mask

    Returns
    -------
    labels : list
        list of ground truth labels
    scores : list
        list of scores for predictions
    """

    # convert to numpy arrays
    arr1 = np.array(msk)
    arr2 = np.array(prd)

    # find contours
    contours1 = cv2.findContours(arr1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours2 = cv2.findContours(arr2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    labels = []
    scores = []

    # Find the intersection of the two contours
    for c1 in contours1:
        if len(c1) < 3:
            # not a polygon
            continue

        p1 = Polygon(c1[:, 0])
        hits = 0
        for c2 in contours2:
            try:
                p2 = Polygon(c2[:, 0])
                if p1.intersects(p2):
                    hits += 1

                    score = p1.intersection(p2).area / p1.union(p2).area
                    labels.append(1)
                    scores.append(score)

            except:
                continue

        if hits == 0:
            # no intersection, add false negative result to list
            labels.append(1)
            scores.append(0.0)

    for c2 in contours2:
        try:
            p2 = Polygon(c2[:, 0])
        except:
            continue

        for c1 in contours1:
            try:
                p1 = Polygon(c1[:, 0])
                if p1.intersects(p2):
                    break

            except:
                continue

        else:
            # no intersection, add false positive result to list
            labels.append(0)
            scores.append(1.0)

    return labels, scores


def compute_iou_scores(
    msk: Image.Image, sft: Image.Image, n_thresholds: int = 11
) -> Tuple[list, list, list]:
    """Computes IOU scores for all individual predicted objects, at different thresholds.
    False positive samples are assigned a score of 1, so the iou_cutoff will assign a True prediction.

    Parameters
    ----------
    msk : Image.Image
        ground truth mask
    sft : Image.Image
        predicted mask with soft edges
    n_thresholds : int
        number of thresholds to use

    Returns
    -------
    labels : list
        list of ground truth labels
    scores : list
        list of IOU scores for predictions
    thresh : list
        list of thresholds used. These threshold are used in the discretization of the softened predicted mask
    """

    thresholds = np.linspace(0.0, 1.0, n_thresholds).tolist()
    labels = []
    scores = []
    thresh = []
    for thr in thresholds:
        tmp = prediction_tools.threshold_prediction(sft, thr)
        ls, ss = get_labels_and_scores(msk, tmp)
        labels.extend(ls)
        scores.extend(ss)
        thresh.extend([thr] * len(ls))

    return labels, scores, thresh


def compute_precision_recall_curve(
    msk: Image.Image, prd: Image.Image, iou_cutoff: float = 0.5, n_thresholds: int = 11
) -> Tuple[list, list, list]:
    """Compute the precision-recall curve for a given prediction.

    Parameters
    ----------
    msk : Image
        ground truth mask
    prd : Image
        prediction with soft edges
    iou_cutoff : float
        intersection over union cutoff, for determining hits
    n_thresholds : int
        number of thresholds to use. These threshold are used in the discretization of prd image

    Returns
    -------
    precision : list
        precision values
    recall : list
        recall values
    thresholds : list
        thresholds
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds).tolist()
    precision = []
    recall = []
    for thr in thresholds:
        tmp = prediction_tools.threshold_prediction(prd, thr)
        labels, scores = get_labels_and_scores(msk, tmp)
        predic = [int(score > iou_cutoff) for score in scores]

        TP = sum([int(l == 1 and p == 1) for l, p in zip(labels, predic)])
        FP = sum([int(l == 0 and p == 1) for l, p in zip(labels, predic)])
        FN = sum([int(l == 1 and p == 0) for l, p in zip(labels, predic)])

        precision.append(TP / (TP + FP) if TP + FP > 0 else 1.0)
        recall.append(TP / (TP + FN) if TP + FN > 0 else 1.0)

    return precision, recall, thresholds


def compute_confusion_matrix(
    lbl_arr: np.ndarray[typing.Any, np.dtype[bool]],
    prd_arr: np.ndarray[typing.Any, np.dtype[bool]],
):
    tp = np.logical_and(lbl_arr == True, prd_arr == True).sum()
    fp = np.logical_and(lbl_arr == False, prd_arr == True).sum()
    tn = np.logical_and(lbl_arr == False, prd_arr == False).sum()
    fn = np.logical_and(lbl_arr == True, prd_arr == False).sum()

    return (
        int(tp),
        int(fp),
        int(tn),
        int(fn),
    )


def compute_metrics():
    metric_fn = dict(
        accuracy=lambda tp, fp, tn, fn: (tp + tn) / (tp + fp + tn + fn),
        precision=lambda tp, fp, tn, fn: tp / (tp + fp),
        recall=lambda tp, fp, tn, fn: tp / (tp + fn),  # sensitivity
        specificity=lambda tp, fp, tn, fn: tn / (tn + fp),
        f1_score=lambda tp, fp, tn, fn: (1 + 1**2)
        * ((tp / (tp + fp)) * (tp / (tp + fn)))
        / (((tp / (tp + fp)) * 1**2) + (tp / (tp + fn))),
        f2_score=lambda tp, fp, tn, fn: (1 + 2**2)
        * ((tp / (tp + fp)) * (tp / (tp + fn)))
        / (((tp / (tp + fp)) * 2**2) + (tp / (tp + fn))),
    )
