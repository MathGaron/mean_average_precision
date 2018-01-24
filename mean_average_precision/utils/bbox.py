# -*- coding: utf-8 -*-
"""
    Bounding box intersection over union calculation.
    Borrowed from pytorch SSD implementation : https://github.com/amdegroot/ssd.pytorch/blob/master/layers/box_utils.py
    and adapted to numpy.
"""
import numpy as np


def intersect_area(box_a, box_b):
    """
    Compute the area of intersection between two rectangular bounding box
    Bounding boxes use corner notation : [x1, y1, x2, y2]
    Args:
      box_a: (np.array) bounding boxes, Shape: [A,4].
      box_b: (np.array) bounding boxes, Shape: [B,4].
    Return:
      np.array intersection area, Shape: [A,B].
    """
    resized_A = box_a[:, np.newaxis, :]
    resized_B = box_b[np.newaxis, :, :]
    max_xy = np.minimum(resized_A[:, :, 2:], resized_B[:, :, 2:])
    min_xy = np.maximum(resized_A[:, :, :2], resized_B[:, :, :2])

    diff_xy = (max_xy - min_xy)
    inter = np.clip(diff_xy, a_min=0, a_max=np.max(diff_xy))
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (np.array) Predicted bounding boxes,    Shape: [n_pred, 4]
        box_b: (np.array) Ground Truth bounding boxes, Shape: [n_gt, 4]
    Return:
        jaccard overlap: (np.array) Shape: [n_pred, n_gt]
    """
    inter = intersect_area(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]))
    area_a = area_a[:, np.newaxis]
    area_b = area_b[np.newaxis, :]
    union = area_a + area_b - inter
    return inter / union