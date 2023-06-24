import pickle
import os
import numpy as np
from shapely.geometry import *
from tqdm import tqdm
import cv2


# Visualization
def visualize(img_path, pts):
    img = cv2.imread(img_path)
    pts = np.array(
        [(int(pts[i]), int(pts[i + 1])) for i in range(0, len(pts), 2)], dtype=np.int32
    )
    img = cv2.polylines(img, [pts], True, (0, 255, 255))
    img = cv2.resize(img, (1024, 1024))
    cv2.imshow("test", img)
    cv2.waitKey(0)


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval_polygon(gt_path, detfile, classname, ovthresh=0.5, use_07_metric=False):
    image_names = sorted(os.listdir(gt_path))

    class_recs = {}
    npos = 0
    num_gt = {}

    for i in range(len(image_names)):
        with open(os.path.join(gt_path, image_names[i]), encoding='utf-8') as f:
            try:
                data = f.readlines()
            except:
                print(i)
            ptses = []
            for line in data:
                pts = line.split('|')[0]
                pts = pts.split(' ')[:-1]
                pts = pts[6:]
                ptses.append(pts)
            ptses = np.array(ptses, dtype=np.int32)
            difficult = np.array([False] * len(ptses)).astype(bool)
            det = np.array([False] * len(ptses)).astype(bool)
            npos = npos + sum(~difficult)
            num_gt[str(i)] = sum(~difficult)
            class_recs[str(i)] = {'ptses': ptses, 'difficult': difficult, 'det': det}


    # read dets
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    # confidence = np.array([float(x[1]) for x in splitlines])

    BB = []
    for x in splitlines:
        bb = np.array([float(z.strip().replace("\n", "")) for z in x[6:]])
        BB.append(bb)

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    # for d in range(nd):
    for d in range(nd):
        R = class_recs[image_ids[d]]

        bb = BB[d]  # mask rcnn
        det_bbox = bb[:]
        pts = [(det_bbox[j], det_bbox[j + 1]) for j in range(0, len(bb), 2)]
        pdet = Polygon(pts)

        ovmax = -np.inf
        BBGT = R["ptses"].astype(float)
        overlaps = np.zeros(BBGT.shape[0])
        for iix in range(BBGT.shape[0]):
            pts = [
                (
                    int(BBGT[iix, j]) ,
                    int(BBGT[iix, j + 1])
                )
                for j in range(0, len(BBGT[0]), 2)
            ]
            pgt = Polygon(pts)
            try:
                sec = pdet.intersection(pgt)
            except Exception as e:
                continue
            inters = sec.area
            uni = pgt.area + pdet.area - inters
            if uni <= 0.00001:
                uni = 0.00001
            overlaps[iix] = inters * 1.0 / uni

        ovmax = np.max(overlaps)
        jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fpp = fp
    tpp = tp
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap, fpp, tpp, image_ids, num_gt
