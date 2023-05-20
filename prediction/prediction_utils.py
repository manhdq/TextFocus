from datetime import datetime

import numpy as np
##TODO: Make this variable dynamic
MAXIMUM_NUM_CLASSES = 2 # 0: Background, 1: Text


def aggregate_prediction_result(img_name,
                        flattened_infer_results,
                        count_results,
                        start_pred_time,
                        ori_pred_save_path,
                        max_focus_rank,
                        ori_image_shape):
    """
    Aggregate infer results (collect reason, count result ...)
    to dump to json file
    """
    if count_results is not None:
        ##TODO: Make this dynamic for dynamic number of categories
        num_text = count_results[0]
    else:
        num_text = -1

    reformated_infer_result = {
        img_name: {
            "ori_image_with_merged_preds_save_path": ori_pred_save_path,
            "max_focus_rank": max_focus_rank,
            "num_text": num_text,
            "predictions": flattened_infer_results,
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
