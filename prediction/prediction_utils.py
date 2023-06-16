'''
Utilities used in inference and demo
'''
from datetime import datetime

import copy
import numpy as np
import cv2
from shapely.geometry import Polygon
##TODO: Make this variable dynamic
MAXIMUM_NUM_CLASSES = 2 # 0: Background, 1: Text


def find_new_size(image_width, image_height, valid_range, base_scale_down):
    '''
    Find new size and new scale of image base on valid range
    '''
    def _find_new_size(image_width, image_height, scale):
        return (round(image_width / scale), round(image_height / scale))
    min_ori_size, max_ori_size = sorted((image_height, image_width))

    # Scale down original image for the first forward pass
    new_size = _find_new_size(image_width, image_height, scale=base_scale_down)
    valid_min_size, valid_max_size = valid_range

    # Scale to valid range
    if min(new_size) > valid_min_size:
        base_scale_down = min_ori_size / valid_min_size
        new_size = _find_new_size(image_width, image_height, scale=base_scale_down)
        if max(new_size) > valid_max_size:
            base_scale_down = max_ori_size / valid_max_size
            new_size = _find_new_size(image_width=image_width,
                                                image_height=image_height,
                                                scale=base_scale_down)
    return new_size, base_scale_down


####################################
##### SCALE N SHIFT COORDINATE #####
####################################

def batch_scale_n_shift_dets(py_preds, scale, left_shift, top_shift):
    '''
    Scale and shift old coordinate to new coordinate
    `py_preds` contains list of lm preds according to `GDN` level
    '''
    _py_preds = []
    for cur_py_preds in py_preds:
        _cur_py_preds = cur_py_preds.copy()
        _cur_py_preds *= scale
        
        _cur_py_preds[..., 0::2] = _cur_py_preds[..., 0::2] + left_shift
        _cur_py_preds[..., 1::2] = _cur_py_preds[..., 1::2] + top_shift
        
        _py_preds.append(_cur_py_preds)
    return _py_preds


####################################
######### EARLY STOP UTILS #########
####################################

class RejectException(Exception):
    '''
    Exception raises
    when early stopping process found rejected face
    '''


##TODO: Delete this
def polygons_area(polygons):
    """
    Shape of polygons: (num_points x 2)
    """
    if len(polygons.shape) == 2:  # num_points x 2
        polygons = polygons[:, None]
    return cv2.contourArea(polygons)


def get_polygon_intersections(cur_polygon, target_polygons):
    inter_list = []
    for target_polygon in target_polygons:
        inter_list.append(cur_polygon.intersection(target_polygon).area)
    inter_array = np.array(inter_list)

    return inter_array


def nms(points_group, confidences, thresh):
    """Modified Python NMS for keypoints prediction"""
    polygons_group = [Polygon(points).buffer(0) for points in points_group]
    areas = np.array([polygons.area for polygons in polygons_group])
    order = confidences.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        inter = get_polygon_intersections(polygons_group[i], [polygons_group[j] for j in order[1:]])
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def get_index(item, idx):
    return item[idx]


def filter_preds(py_preds,
                confidences,
                top_k_before_nms,
                nms_threshold,
                top_k_after_nms):
                
    # Get max score and corresponding class for each prediction
    # Keep top-k before NMS
    order = confidences.argsort()[::-1][:top_k_before_nms]
    ##TODO: Number of iter not dynamic
    py_preds[0], py_preds[1], py_preds[2], py_preds[3], confidences = \
        map(get_index, [*py_preds, confidences], [order]*5)


    # Do NMS
    assert len(py_preds[-1]) == len(confidences)
    if len(py_preds[-1]) > 1:
        keep = nms(py_preds[-1], confidences, nms_threshold)
        py_preds = [py_pred[keep] for py_pred in py_preds]
        confidences = confidences[keep]

    # Keep top-K after NMS
    py_preds = [py_pred[:top_k_after_nms] for py_pred in py_preds]
    confidences = confidences[:top_k_after_nms]
    
    return py_preds, confidences


def aggregate_prediction_result(img_name,
                            flattened_prediction_results,
                            start_pred_time,
                            ori_pred_save_path,
                            max_focus_rank,
                            ori_image_shape):
    reformated_infer_result = {
        img_name: {
            "ori_image_with_merged_preds_save_path": ori_pred_save_path,
            "max_focus_rank": max_focus_rank,
            "predictions": flattened_prediction_results,
            "ori_image_shape": ori_image_shape,
            "prediction_time": (datetime.now() - start_pred_time).total_seconds(),
        }
    }

    return reformated_infer_result


def log_info(start_pred_time, img_name, rank=None):
    '''
    Log prediction time to terminal
    '''
    infer_time = datetime.now() - start_pred_time
    print(f'{infer_time} | {img_name}')