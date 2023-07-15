import os
import torch
from .map import mAPScores
from .metrics import *
from .cal_recall.script import  cal_recall_precison_f1


def cal_text_score(texts, gt_texts, training_masks, running_metric_text):
    training_masks = training_masks.data.cpu().numpy()
    pred_text = torch.sigmoid(texts).data.cpu().numpy() * training_masks
    pred_text[pred_text <= 0.5] = 0
    pred_text[pred_text > 0.5] = 1
    pred_text = pred_text.astype(np.int32)
    gt_text = gt_texts.data.cpu().numpy() * training_masks
    gt_text = gt_text.astype(np.int32)
    running_metric_text.update(gt_text, pred_text)
    score_text, _ = running_metric_text.get_scores()
    return score_text


def cal_kernel_score(kernel, gt_kernel, gt_texts, training_masks, running_metric_kernel):
    mask = (gt_texts * training_masks.float()).data.cpu().numpy()
    pred_kernel = torch.sigmoid(kernel).data.cpu().numpy()
    pred_kernel[pred_kernel <= 0.5] = 0
    pred_kernel[pred_kernel > 0.5] = 1
    pred_kernel = (pred_kernel * mask).astype(np.int32)
    gt_kernel = gt_kernel.data.cpu().numpy()
    gt_kernel = (gt_kernel * mask).astype(np.int32)
    running_metric_kernel.update(gt_kernel, pred_kernel)
    score_kernel, _ = running_metric_kernel.get_scores()
    return score_kernel


def cal_focus_score(focus_preds, focus_gt, running_metric_focus):
    focus_gt = focus_gt.data.cpu().numpy()
    training_masks = (focus_gt != -1).astype(int)
    focus_gt = (focus_gt == 1).astype(int) * training_masks
    _, H, W = focus_gt.shape

    focus_preds = torch.softmax(focus_preds, dim=1).data.cpu().numpy()[:, 1]
    focus_preds[focus_preds <=0.5] = 0
    focus_preds[focus_preds >0.5] = 1
    focus_preds = focus_preds.reshape((focus_preds.shape[0], H, W))
    focus_preds = (focus_preds * training_masks).astype(int)

    running_metric_focus.update(focus_gt, focus_preds)
    score_focus, _ = running_metric_focus.get_scores()
    return score_focus
    

def get_metric(config):
    metric = mAPScores(
        ann_dir= os.path.join(config.data_root, "gt", config.val_subroot),
        img_dir=os.path.join(config.data_root, "Images", config.val_subroot)
    )

    return metric
