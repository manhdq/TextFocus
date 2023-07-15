# -*- coding: utf-8 -*-
# @Time    : 2019/9/8 14:18
# @Author  : zhoujun
import os
import cv2
import torch
import time
import subprocess
import numpy as np

from .pypse import pse_py
from .kmeans import km

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def decode(preds, scale=1, threshold=0.7311, min_area=5):
    """
    在输出上使用sigmoid 将值转换为置信度，并使用阈值来进行文字和背景的区分
    :param preds: 网络输出
    :param scale: 网络的scale
    :param threshold: sigmoid的阈值
    :return: 最后的输出图和文本框
    """
    from .pse import pse_cpp, get_points, get_num
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    score = preds[0].astype(np.float32)
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel
    similarity_vectors = preds[2:].transpose((1, 2, 0))

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < min_area:
            continue
        label_values.append(label_idx)

    pred = pse_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, 0.8)
    pred = pred.reshape(text.shape)

    bbox_list = []
    label_points = get_points(pred, score, label_num)
    for label_value, label_point in label_points.items():
        if label_value not in label_values:
            continue
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < 100 / (scale * scale):
            continue

        if score_i < 0.93:
            continue

        rect = cv2.minAreaRect(points)
        bbox = cv2.boxPoints(rect)
        bbox_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])
    return pred, np.array(bbox_list)


def decode_dice(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label_num == label_idx)).transpose((1, 0))[:, ::-1]

        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)

        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))

        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)

def decode_clip(preds, scale=1, threshold=0.7311, min_area=5):
    import pyclipper
    import numpy as np
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    bbox_list = []
    for label_idx in range(1, label_num):
        points = np.array(np.where(label == label_idx)).transpose((1, 0))[:, ::-1]
        if points.shape[0] < min_area:
            continue
        
        rect = cv2.minAreaRect(points)
        poly = cv2.boxPoints(rect).astype(int)
        print(poly.shape)

        d_i = cv2.contourArea(poly) * 1.5 / cv2.arcLength(poly, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(d_i))
        print(shrinked_poly.shape)
        exit()
        if shrinked_poly.size == 0:
            continue
        rect = cv2.minAreaRect(shrinked_poly)
        shrinked_poly = cv2.boxPoints(rect).astype(int)
        if cv2.contourArea(shrinked_poly) < 800 / (scale * scale):
            continue

        bbox_list.append([shrinked_poly[1], shrinked_poly[2], shrinked_poly[3], shrinked_poly[0]])
    return label, np.array(bbox_list)


def norm2(x, axis=None):
    if axis:
        return np.sqrt(np.sum(x**2, axis=axis))
    return np.sqrt(np.sum(x**2))


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


# def decode_polys_clip(preds, num_points=4, scale=1, threshold=0.7311, min_area=5, approx_factor=0.004):
#     import numpy as np
#     preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
#     preds = preds.detach().cpu().numpy()
#     text = preds[0] > threshold  # text
#     kernel = (preds[1] > threshold) * text  # kernel

#     label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
#     points_list = []
#     for label_idx in range(1, label_num):
#         mask = np.array(label == label_idx).astype(np.uint8)
#         # Sort the contours by area in descending order
#         contours, _ = cv2.findContours(
#             mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
#         )
#         print(cv2.arcLength(contours[0], True))
#         exit()
#         contours = sorted(contours, key=cv2.contourArea, reverse=True)
#         epsilon = approx_factor * cv2.arcLength(contours[0], True)
#         approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
#         points = split_edge_sequence(approx, num_points)
#         points_list.append(points)
#     return label, np.array(points_list)


def decode_polys_clip(preds, num_points=4, scale=1, threshold=0.7311, min_area=5, approx_factor=0.004):
    import pyclipper
    from .pse import pse_cpp, get_points, get_num
    import numpy as np
    preds[:2, :, :] = torch.sigmoid(preds[:2, :, :])
    preds = preds.detach().cpu().numpy()
    score = preds[0].astype(np.float32)
    text = preds[0] > threshold  # text
    kernel = (preds[1] > threshold) * text  # kernel
    similarity_vectors = preds[2:].transpose((1, 2, 0))

    label_num, label = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=4)
    label_values = []
    label_sum = get_num(label, label_num)
    for label_idx in range(1, label_num):
        if label_sum[label_idx] < min_area:
            continue
        label_values.append(label_idx)

    pred = pse_cpp(text.astype(np.uint8), similarity_vectors, label, label_num, 0.6)
    pred = pred.reshape(text.shape)

    points_list = []
    label_points = get_points(pred, score, label_num)
    for label_value, label_point in label_points.items():
        if label_value not in label_values:
            continue
        score_i = label_point[0]
        label_point = label_point[2:]
        points = np.array(label_point, dtype=int).reshape(-1, 2)

        if points.shape[0] < 100 / (scale * scale):
            continue

        if score_i < 0.75:  ##TODO: What this??
            continue

        mask_re = np.zeros_like(pred).astype(np.uint8)
        mask_re = cv2.polylines(mask_re, [points], True, 1, 2)
        contours, _ = cv2.findContours(
            mask_re.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # print(cv2.arcLength(points[:, None], True))
        # print(cv2.arcLength(contours[0], True))
        ##TODO: priority. Modify approx factor for saving time
        epsilon = approx_factor * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        points = split_edge_sequence(approx, num_points)[:-1].astype(int)

        d_i = cv2.contourArea(points) * 0.4 / cv2.arcLength(points, True)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_points = np.array(pco.Execute(d_i))[0]
        
        mask_re = np.zeros_like(pred).astype(np.uint8)
        mask_re = cv2.polylines(mask_re, [shrinked_points], True, 1, 2)
        contours, _ = cv2.findContours(
            mask_re.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        # print(cv2.arcLength(points[:, None], True))
        # print(cv2.arcLength(contours[0], True))
        ##TODO: priority. Modify approx factor for saving time
        epsilon = approx_factor * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True).reshape((-1, 2))
        points = split_edge_sequence(approx, num_points)[:-1].astype(int)

        # Select exact number of points


        points_list.append(points)

        # print(cv2.arcLength(points[:, None], True))
        # print(cv2.arcLength(contours[0], True))
        # exit()
        # epsilon = approx_factor * cv2.arcLength(points[:, None], True)
        # approx = cv2.approxPolyDP(points[:, None], epsilon, True).reshape((-1, 2))
        # points = split_edge_sequence(approx, num_points)
        # points_list.append(points)
        # test = np.zeros_like(pred).astype(np.uint8)
        # test = cv2.polylines(np.stack([test]*3, -1), [points], True, (255, 0, 0), 2)
        # cv2.imwrite("test.jpg", test)
        # print(test.shape)
        # exit()
        # print(points)

        # rect = cv2.minAreaRect(points)
        # bbox = cv2.boxPoints(rect)
        # points_list.append([bbox[1], bbox[2], bbox[3], bbox[0]])

    return pred, np.array(points_list)