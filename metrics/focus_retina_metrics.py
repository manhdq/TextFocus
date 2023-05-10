'''
FocusRetina Metrics
'''
from math import ceil

import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .metrics_lib import (Evaluator, BoundingBox, BoundingBoxes, BBType,
                          BBFormat, CoordinatesType, MethodAveragePrecision)
from utils.misc import LABEL2NAME


def cal_img_metrics(gts, preds, average='binary'):
    acc = accuracy_score(gts, preds)
    precision = precision_score(gts, preds, average=average)
    recall = recall_score(gts, preds, average=average)
    f1 = f1_score(gts, preds, average=average)
    return acc, precision, recall, f1


class FocusRetinaMetricsCalculator():
    '''
    FocusRetina Metrics
    '''

    def __init__(self, iou_threshold, mask_conf_threshold, stride):
        self.evaluator = Evaluator()
        self.bb_format = BBFormat.XYX2Y2
        self.type_coordinate = CoordinatesType.Absolute
        self.method_cal_ap = MethodAveragePrecision.EveryPointInterpolation
        self.iou_threshold = iou_threshold
        self.mask_conf_threshold = mask_conf_threshold
        self.stride = stride

    @staticmethod
    def _cal_mask_iou_dice(all_mask_preds, all_mask_gts):
        '''
        Calculate intersection over union and dice
        between mask_pred and mask_gt of all images
        '''
        iou_scores, dice_scores = [], []
        for mask_pred, mask_gt in zip(all_mask_preds, all_mask_gts):
            sum_mask = mask_pred + mask_gt
            # Index 0 means position if condition is true
            intersection = np.where(sum_mask == 2)[0]
            union = np.where(sum_mask != 0)[0]
            if len(union) != 0:
                iou_score = len(intersection) / len(union)
                dice_score = 2 * len(intersection) / \
                    (len(union) + len(intersection))
            else:
                # continue
                iou_score = 1
                dice_score = 1
            iou_scores.append(iou_score)
            dice_scores.append(dice_score)
        return sum(iou_scores) / len(iou_scores), sum(dice_scores) / len(dice_scores)

    @staticmethod
    def _cal_mask_diff(all_mask_preds, all_mask_gts):
        diffs = []
        max_diff = 0
        min_diff = np.inf
        for mask_img_pred, mask_img_gt in zip(all_mask_preds, all_mask_gts):
            mask_img_pred, mask_img_gt = mask_img_pred * 255, mask_img_gt * 255
            pred_cnts, _ = cv2.findContours(
                mask_img_pred.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            gt_cnts, _ = cv2.findContours(
                mask_img_gt.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            diff = np.abs(len(pred_cnts) - len(gt_cnts))
            max_diff = diff if diff > max_diff else max_diff
            min_diff = diff if diff < min_diff else min_diff
            diffs.append(diff)
        return sum(diffs) / len(diffs), max_diff, min_diff

    def _cal_map_score(self, bbox_list):
        '''
        Calculate mAP over all bboxes of all images in the dataset
        '''
        metrics_all_classes = self.evaluator.GetPascalVOCMetrics(
            boundingboxes=bbox_list,
            IOUThreshold=self.iou_threshold,
            method=self.method_cal_ap)

        ap_scores = dict()
        for metrics_per_class in metrics_all_classes:
            ap_scores[metrics_per_class['class']] = metrics_per_class['AP']
        map_score = sum(ap_scores.values()) / len(ap_scores.values())
        ap_scores = {
            LABEL2NAME[int(key)]: ap_scores[key]
            for key in ap_scores.keys()
        }
        return map_score, ap_scores

    def __call__(self, all_preds, all_gts, img_size):
        assert len(all_preds) == len(all_gts), \
            'Len of preds is not equal to len of gts'
        print(f'Calculate metrics on {len(all_preds)} images...')

        mask_size = (ceil(img_size[0] / self.stride),
                     ceil(img_size[1] / self.stride))

        all_bboxes = BoundingBoxes()
        all_mask_preds = []
        all_mask_gts = []
        for i, (img_preds, img_gts) in tqdm(enumerate(zip(all_preds, all_gts))):
            det_preds, foc_preds = img_preds
            # Prepare pred focus mask
            square_foc_preds = foc_preds[1].reshape(mask_size)
            all_mask_preds.append(
                np.where(square_foc_preds > self.mask_conf_threshold, 1, 0))
            # Prepare pred detections
            bb_preds, conf_preds, _, cls_preds = \
                det_preds[:, :4], det_preds[:, 4:5], det_preds[:, 5:15], det_preds[:, 15:]

            # Add bbox prediction to all_bboxes list
            for bb_pred, conf_pred, cls_pred in zip(bb_preds, conf_preds, cls_preds):
                all_bboxes.addBoundingBox(
                    BoundingBox(imageName=f'img_{i}',
                                classId=str(int(cls_pred)),
                                classConfidence=conf_pred,
                                x=float(bb_pred[0]),
                                y=float(bb_pred[1]),
                                w=float(bb_pred[2]),
                                h=float(bb_pred[3]),
                                typeCoordinates=self.type_coordinate,
                                bbType=BBType.Detected,
                                format=self.bb_format,
                                imgSize=img_size)
                )

            det_gts, foc_gts = img_gts
            # Prepare gt focus mask
            all_mask_gts.append(foc_gts.reshape(mask_size))
            # Prepare gt detections
            bb_gts, _, cls_gts = \
                det_gts[:, :4], det_gts[:, 4:14], det_gts[:, 14:]

            # Add bbox gt to all_bboxes list
            for bb_gt, cls_gt in zip(bb_gts, cls_gts):
                all_bboxes.addBoundingBox(
                    BoundingBox(imageName=f'img_{i}',
                                classId=str(int(cls_gt)),
                                x=float(bb_gt[0] * img_size[0]),
                                y=float(bb_gt[1] * img_size[1]),
                                w=float(bb_gt[2] * img_size[0]),
                                h=float(bb_gt[3] * img_size[1]),
                                typeCoordinates=self.type_coordinate,
                                bbType=BBType.GroundTruth,
                                format=self.bb_format,
                                imgSize=img_size)
                )

        # Calculate metrics
        map_score, ap_scores = self._cal_map_score(all_bboxes)
        mask_iou_score, mask_dice_score = self._cal_mask_iou_dice(
            all_mask_preds, all_mask_gts)
        n_mask_diff, max_mask_diff, min_mask_diff = \
            self._cal_mask_diff(all_mask_preds, all_mask_gts)

        focus_retina_metrics = {
            'detection': {
                'ap': ap_scores,
                'map': map_score
            },
            'focus': {
                'iou': mask_iou_score,
                'dice': mask_dice_score,
                'diff': n_mask_diff,
                'max_mask_diff': max_mask_diff,
                'min_mask_diff': min_mask_diff
            }
        }
        return focus_retina_metrics
