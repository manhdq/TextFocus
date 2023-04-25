import random
import os

import numpy as np
import torch
from ptflops import get_model_complexity_info

from .box_utils import decode_np, decode_landm_np, decode_batch_np, decode_landm_batch_np
from .nms import nms


LABEL2NAME = {
    0: 'BG',
    1: 'MR',
    2: 'no-MR',
    3: 'Fake-2D',
    4: 'Fake-3D',
    5: 'Human'
}


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu vars
    random.seed(seed_value)  # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def show_params_gflops(model, size, print_layer=True, input_constructor=None, device='cuda:0'):
    ''' Calculate and show params gflops of model'''
    with torch.cuda.device(device):
        macs, params = get_model_complexity_info(model,
                                                 size,
                                                 print_per_layer_stat=print_layer,
                                                 input_constructor=input_constructor)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def get_index(item, idx):
    return item[idx]


def _filter_preds(max_classes,
                  max_scores,
                  boxes,
                  landms,
                  top_k_before_nms,
                  nms_threshold,
                  top_k_after_nms,
                  nms_per_class):

    # Ignore background
    inds = max_classes != 0
    boxes, landms, max_scores, max_classes = \
        map(get_index, [boxes, landms, max_scores, max_classes], [inds]*4)

    # Keep top-K before NMS
    order = max_scores.argsort()[::-1][:top_k_before_nms]
    boxes, landms, max_scores, max_classes = \
        map(get_index, [boxes, landms, max_scores, max_classes], [order]*4)

    # Do NMS
    dets = np.hstack((boxes, max_scores[:, np.newaxis])).astype(np.float32, copy=False)
    if nms_per_class:
        all_classes = np.unique(max_classes)
        all_dets = np.empty((0, 5)).astype(dets.dtype)
        all_landms = np.empty((0, 10)).astype(landms.dtype)
        all_max_classes = np.empty((0,)).astype(max_classes.dtype)
        for class_id in all_classes:
            class_inds = max_classes == class_id
            class_dets = dets[class_inds]
            class_landms = landms[class_inds]
            class_max_classes = max_classes[class_inds]
            if len(class_dets) > 1:
                keep = nms(class_dets, nms_threshold)
                class_dets = class_dets[keep]
                class_landms = class_landms[keep]
                class_max_classes = class_max_classes[keep]
            all_dets = np.concatenate((all_dets, class_dets), axis=0)
            all_landms = np.concatenate((all_landms, class_landms), axis=0)
            all_max_classes = np.concatenate((all_max_classes, class_max_classes), axis=0)
        dets = all_dets
        landms = all_landms
        max_classes = all_max_classes
        order = dets[:, -1].argsort()[::-1]
        dets = dets[order]
        landms = landms[order]
        max_classes = max_classes[order]
    else:
        if len(dets) > 1:
            keep = nms(dets, nms_threshold)
            dets = dets[keep]
            landms = landms[keep]
            max_classes = max_classes[keep]

    # Keep top-K after NMS
    dets = dets[:top_k_after_nms]
    landms = landms[:top_k_after_nms]
    max_classes = max_classes[:top_k_after_nms]

    dets = np.concatenate((dets, landms, max_classes[:, np.newaxis]), axis=1)

    return dets


def filter_preds(cls_pred,
                 loc_pred,
                 lm_pred,
                 box_scale,
                 lm_scale,
                 priors,
                 variance,
                 top_k_before_nms,
                 nms_threshold,
                 top_k_after_nms,
                 nms_per_class):

    boxes = decode_np(loc_pred, priors, variance)
    boxes *= box_scale

    landms = decode_landm_np(lm_pred, priors, variance)
    landms *= lm_scale

    # Get max score and corresponding class for each prediction (bbox)
    max_classes = np.argmax(cls_pred, axis=-1)
    max_scores = np.max(cls_pred, axis=-1) #TODO must optimize this operator

    dets = _filter_preds(max_classes=max_classes,
                         max_scores=max_scores,
                         boxes=boxes,
                         landms=landms,
                         top_k_before_nms=top_k_before_nms,
                         nms_threshold=nms_threshold,
                         top_k_after_nms=top_k_after_nms,
                         nms_per_class=nms_per_class)

    return dets


def filter_batch_preds(cls_preds,
                       loc_preds,
                       lm_preds,
                       box_scale,
                       lm_scale,
                       priors,
                       variance,
                       top_k_before_nms,
                       nms_threshold,
                       top_k_after_nms,
                       nms_per_class):

    batch_boxes = decode_batch_np(loc_preds, priors, variance)
    batch_boxes *= box_scale

    batch_landms = decode_landm_batch_np(lm_preds, priors, variance)
    batch_landms *= lm_scale

    # Get max score and corresponding class for each prediction (bbox)
    batch_max_classes = np.argmax(cls_preds, axis=-1)
    batch_max_scores = np.max(cls_preds, axis=-1) #TODO must optimize this operator

    batch_dets = []
    for max_classes, max_scores, boxes, landms in zip(batch_max_classes, batch_max_scores, batch_boxes, batch_landms):
        dets = _filter_preds(max_classes=max_classes,
                             max_scores=max_scores,
                             boxes=boxes,
                             landms=landms,
                             top_k_before_nms=top_k_before_nms,
                             nms_threshold=nms_threshold,
                             top_k_after_nms=top_k_after_nms,
                             nms_per_class=nms_per_class)
        batch_dets.append(dets)

    return batch_dets
