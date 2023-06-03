import numpy as np
import errno
import os
import cv2
import math
from shapely.geometry import Polygon
from scipy import ndimage as ndimg

from cfglib.config import config as cfg


def to_device(*tensors):
    if len(tensors) < 2:
        return tensors[0].to(cfg.device, non_blocking=True)
    return (t.to(cfg.device, non_blocking=True) for t in tensors)


def mkdirs(newdir):
    """
    make directory with parent path
    :param newdir: target path
    """
    try:
        if not os.path.exists(newdir):
            os.makedirs(newdir)
    except OSError as err:
        # Reraise the error unless it's about an already existing directory
        if err.errno != errno.EEXIST or not os.path.isdir(newdir):
            raise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x**2, axis=axis))
    return np.sqrt(np.sum(x**2))


def cos(p1, p2):
    return (p1 * p2).sum() / (norm2(p1) * norm2(p2))


def split_edge_sequence(points, n_parts):
    pts_num = points.shape[0]
    long_edge = [(i, (i + 1) % pts_num) for i in range(pts_num)]
    edge_length = [norm2(points[e1] - points[e2]) for e1, e2 in long_edge]
    point_cumsum = np.cumsum([0] + edge_length)
    total_length = sum(edge_length)
    length_per_part = total_length / n_parts

    cur_node = 0  # first point
    splitted_result = []

    for i in range(1, n_parts):
        cur_end = i * length_per_part

        while cur_end > point_cumsum[cur_node + 1]:
            cur_node += 1

        e1, e2 = long_edge[cur_node]
        e1, e2 = points[e1], points[e2]

        # start_point = points[long_edge[cur_node]]
        end_shift = cur_end - point_cumsum[cur_node]
        ratio = end_shift / edge_length[cur_node]
        new_point = e1 + ratio * (e2 - e1)
        splitted_result.append(new_point)

    # Add first and last point
    p_first = points[long_edge[0][0]]
    p_last = points[long_edge[-1][1]]
    splitted_result = [p_first] + splitted_result + [p_last]
    return np.stack(splitted_result)


def get_sample_point(text_mask, num_points, approx_factor, scales=None):
    # Get sample point in contours
    contours, _ = cv2.findContours(
        text_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    epsilon = approx_factor * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
    # approx = contours[0].reshape((-1, 2))
    if scales is None:
        ctrl_points = split_edge_sequence(approx, num_points)
    else:
        ctrl_points = split_edge_sequence(approx * scales, num_points)
    ctrl_points = np.array(ctrl_points[:num_points, :]).astype(np.int32)

    return ctrl_points