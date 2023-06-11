# -*- coding: utf-8 -*-
import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable


def get_node_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().float()
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1

    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        ##TODO: Understanding `torch.nn.functional.grid_sample` func
        gcn_feature[ind == i] = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
    return gcn_feature