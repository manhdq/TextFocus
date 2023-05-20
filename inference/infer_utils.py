'''
Utilities used in inference
'''
from datetime import datetime

import numpy as np
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

    # Scale to valid_range
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

def batch_scale_n_shift(batch_coord, scale, left_shift, top_shift):
    '''
    Scale and shift old coordinate to new coordinate
    batch_coord = [:, (x1, y1, x2, y2, x3, y3, ...)]
    '''
    _batch_coord = batch_coord.copy().astype(np.float32)

    # Scale coordinates
    _batch_coord *= scale

    # Shift coordinates
    _batch_coord[:, 0::2] = _batch_coord[:, 0::2] + left_shift
    _batch_coord[:, 1::2] = _batch_coord[:, 1::2] + top_shift

    return _batch_coord


def batch_scale_n_shift_dets(dets, scale, left_shift, top_shift):
    '''
    Scale and shift old coordinate to new coordinate
    `dets` contains bbox coordinate, conf, lm coordinate, bbox class
    '''
    _dets = dets.copy()
    _dets[:, :4] = batch_scale_n_shift(batch_coord=_dets[:, :4],
                                    scale=scale,
                                    left_shift=left_shift,
                                    top_shift=top_shift)
    _dets[:, 5:15] = batch_scale_n_shift(batch_coord=_dets[:, 5:15],
                                         scale=scale,
                                         left_shift=left_shift,
                                         top_shift=top_shift)

    return _dets


####################################
######### EARLY STOP UTILS #########
####################################

class RejectException(Exception):
    '''
    Exception raises
    when early stopping process found rejected face
    '''


def aggregate_infer_result(idx,
                        item_id,
                        flattened_infer_results,
                        count_results,
                        reason,
                        ori_gt_save_path,
                        ori_crop_face_info,
                        max_focus_rank,
                        ori_image_shape):
    '''
    Aggregate infer results (collect decision, reason, count result ...)
    to dump to json file
    '''
    pass


def aggregate_infer_result(idx,
                        item_id,
                        flattened_infer_results,
                        count_results,
                        start_pred_time,
                        ori_gt_save_path,
                        ori_pred_save_path,
                        ori_crop_face_info,
                        max_focus_rank,
                        ori_image_shape):
    """
    Aggregate infer results (collect reason, count result ...)
    to dump to json file
    """
    if count_results is not None:
        ##TODO: Make this dynamic for dynamic number of categories
        num_text = map(int, count_results)
    else:
        num_text = -1

    reformated_infer_result = {
        item_id: {
            "ori_image_with_merged_preds_save_path": ori_pred_save_path,
            "ori_image_with_gts_save_path": ori_gt_save_path,
            "index": idx,
            "max_focus_rank": max_focus_rank,
            "num_text": num_text,
            "predictions": flattened_infer_results,
            "ori_image_shape": ori_image_shape,
            "prediction_time": (datetime.now() - start_pred_time).total_seconds(),
        }
    }

    return reformated_infer_result


def log_info(idx, n_batches, start_pred_time, item_id, rank=None):
    '''
    Log infer time to terminal
    '''
    infer_time = datetime.now() - start_pred_time
    print(f'({idx + 1}/{n_batches}) {infer_time} | {item_id}')


def extract_pred_matrix(dets):
    num_dets = dets.shape[0]
    pred_matrix = np.zeros((num_dets, MAXIMUM_NUM_CLASSES))
    if num_dets > 0:
        for idx, bbox in enumerate(dets):
            pred_matrix[idx][int(bbox[-1])] == 4

    return pred_matrix