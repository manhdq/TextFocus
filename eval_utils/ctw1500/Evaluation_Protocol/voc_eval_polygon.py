import os
import numpy as np
from tqdm import tqdm
import cv2
from shapely.geometry import *

try:
    import cPickle
except:
    import _pickle as cPickle

image_path = 'd:\\TEXT\\TextFocus\\EvalTest\\test_images'
curr = __file__
def parse_file_txt(filename):
    with open(filename.strip(), "r") as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(" ")
            cors = [int(item) for item in cors]
            obj_struct = {}
            obj_struct["image_id"] = cors[0]
            obj_struct["ori_bbox"] = cors[1:5]
            obj_struct["offset_polygon"] = cors[5:]
            obj_struct["pts"] = [
                (
                    obj_struct["ori_bbox"][0] + obj_struct["offset_polygon"][i],
                    obj_struct["ori_bbox"][1] + obj_struct["offset_polygon"][i + 1],
                )
                for i in range(0, len(obj_struct["offset_polygon"]), 2)
            ]
            objects.append(obj_struct)
    return objects


def check_exist_pts(object):
    return len(object.get("pts")) > 0


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


def voc_eval_polygon(detpath, annopath, ovthresh=0.5, use_07_metric=False):
    list_det_files = sorted(os.listdir(detpath))
    num_files = len(list_det_files)
    tp = np.zeros(num_files)
    fp = np.zeros(num_files)

    for i in range(len(list_det_files)):
        det_file = list_det_files[i]
        name = det_file.split('.')[0] + '.jpg'
        img_file = os.path.abspath(os.path.join(image_path, name))
        print(img_file)
        img = cv2.imread(img_file)
        print(img.shape)
        det = parse_file_txt(os.path.join(detpath, det_file))
        gt = parse_file_txt(os.path.join(annopath, det_file))
        detect = [False] * len(det)
        overlap = np.zeros(len(det))
        for j in range(len(det)):
            if check_exist_pts(det[j]):
                det_polygon = Polygon(det[j].get("pts"))
                # Check if exists both det poly and gt poly
                if check_exist_pts(gt[j]):
                    try:
                        gt_polygon = Polygon(gt[j].get("pts"))
                    except:
                        print(det_file)
                    # print(np.array(det[j].get('pts')))
                   
                    img = cv2.polylines(img, [np.array(det[j].get('pts'))], True, (255, 0, 0))
                    img = cv2.polylines(img, [np.array(gt[j].get('pts'))], True, (0, 0, 255))
                    inter = det_polygon.intersection(gt_polygon).area
                    uni = det_polygon.area + gt_polygon.area - inter
                    if uni <= 0.00001:
                        uni = 0.00001
                    overlap[j] = inter * 1.0 / uni
                else:
                    print("Detect object not in ground truth")
            else:
                print("Doesn't have any polygon is detected")

        # cv2.imshow('test', img)
        # cv2.waitKey(0)

        overMax = np.max(overlap)
        indexMax = np.argmax(overlap)

        if overMax > ovthresh:
            if not detect[indexMax]:
                tp[i] = 1
                detect[indexMax] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    fpp = fp
    tpp = tp
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(num_files)
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return rec, prec, ap, fpp, tpp, num_files


if __name__ == "__main__":
    detpath = "D:\\TEXT\\TextFocus\\EvalTest\\annotations"
    annopath = "D:\\TEXT\\TextFocus\\EvalTest\\detections"

    rec, prec, ap, fpp, tpp, num_files = voc_eval_polygon(detpath, annopath, ovthresh=0.8, use_07_metric=False)
    
    print(rec)
    print(prec)
    print(ap)
    print(fpp)
    print(tpp)
    print(num_files)
